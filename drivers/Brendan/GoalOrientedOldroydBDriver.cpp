//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "ErrorIndicator.h"
#include "GMGSolver.h"
#include "OldroydBFormulationUW.h"
// #include "H1ProjectionFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PreviousSolutionFunction.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"


#include "CamelliaDebugUtility.h"


using namespace Camellia;

class TopLidBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
};

class CylinderBoundary : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundary(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 5e-1; // be generous b/c dealing with parametric curve
    if ( abs(x) > _radius + 1e-12 )
    // added this exception for the half-space geometries
    {
      return false;
    }
    else {
      return ( sqrt(x*x+y*y) < _radius+tol );
    }
  }
};

class CylinderBoundaryGenerous : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundaryGenerous(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-1; // be generous b/c dealing with parametric curve
    return (sqrt(x*x+y*y) < _radius+tol);
  }
};

class CylinderBoundaryExtreme : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundaryExtreme(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 5e-10; // be generous b/c dealing with parametric curve
    return (sqrt(x*x+y*y) < _radius+tol);
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction<double> {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    if ( (abs(x) < _eps) ) {   // top left
      return x / _eps;
    }
    else if ( abs(1.0-x) < _eps) {     // top right
      return (1.0-x) / _eps;
    }
    else {     // top middle
      return 1;
    }
  }
};

class ParabolicInflowFunction_U1 : public SimpleFunction<double> {
  double _height; // height of channel
public:
  ParabolicInflowFunction_U1(double height) {
    _height = height;
  }
  double value(double x, double y) {
    return 3.0/2.0*(1.0-pow(y/_height,2)); // chosen so that AVERAGE inflow velocity is 1.0
  }
};

class ParabolicInflowFunction_Tun : public SimpleFunction<double> {
  double _i, _j; // index numbers
  double _height; // height of channel
  double _muP; // polymeric viscosity
  double _lambda; // relaxation time
public:
  ParabolicInflowFunction_Tun(double height, double muP, double lambda, int i, int j) {
    _i = i;
    _j = j;
    _height = height;
    _muP = muP;
    _lambda = lambda;
  }
  double value(double x, double y) {
    if (_i == 1 && _j == 1)
      return -3.0/2.0*(1.0-pow(y/_height,2))*(18.0*_muP*_lambda*pow(y/(_height*_height),2));
    else if ((_i == 1 && _j == 2) || (_i == 2 && _j == 1))
      return -3.0/2.0*(1.0-pow(y/_height,2))*(-3.0*_muP*y/(_height*_height));
    else if (_i == 2 && _j == 2)
      return 0.0;
    else
      cout << "ERROR: Indices not currently supported\n";
    return Teuchos::null;
  }
};

int sgn(double val) {
  if (val > 0) return  1;
  if (val < 0) return -1;
  return 0;
}

template <typename Scalar>
class BoundaryOrientedErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  SpatialFilterPtr _spatialFilter;
public:
  BoundaryOrientedErrorIndicator(SolutionPtr soln, SpatialFilterPtr spatialFilter) : ErrorIndicator(soln->mesh())
  {
    _solution = soln;
    _spatialFilter = spatialFilter;
  }

  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {

    const map<GlobalIndexType, double>* rankLocalEnergyError = &_solution->rankLocalEnergyError();
    // square roots have already been taken
    bool energyErrorIsSquared = false;

    _localErrorMeasures.clear();
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      double error = energyErrorIsSquared ? sqrt(entry.second) : entry.second;
      _localErrorMeasures[cellID] = error;
    }

    // calculate max
    double localMax;
    double globalMax;
    for (auto measureEntry : _localErrorMeasures) {
      double cellError = measureEntry.second;
      localMax = max(localMax,cellError);
    }
    _mesh->Comm()->MaxAll(&localMax, &globalMax, 1);

    // add loop through elements to refine elements on the cylinder boundary
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      vector< vector<double> > verticesOfElement = _mesh->verticesForCell(cellID);
      for (int vertexIndex=0; vertexIndex<verticesOfElement.size(); vertexIndex++)
      {
        if (_spatialFilter->matchesPoint(verticesOfElement[vertexIndex][0], verticesOfElement[vertexIndex][1]))
        {
          _localErrorMeasures[cellID] = globalMax;
        }
      }
    }

  }
};

template <typename Scalar>
class GoalOrientedErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  int _cubatureDegreeEnrichment;
public:
  GoalOrientedErrorIndicator(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
  {
    _solution = soln;
    _dualSolnResidualFunction = dualSolnResidualFunction;
    _cubatureDegreeEnrichment = cubatureDegreeEnrichment;
  }

  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {

    const map<GlobalIndexType, double>* rankLocalEnergyError = &_solution->rankLocalEnergyError();
    // square roots have already been taken
    bool energyErrorIsSquared = false;

    _localErrorMeasures.clear();
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      double residual = energyErrorIsSquared ? sqrt(entry.second) : entry.second;
      double dualresidual = sqrt(_dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment));
      _localErrorMeasures[cellID] = residual*dualresidual;
    }

    // calculate max
    double localMax;
    double globalMax;
    for (auto measureEntry : _localErrorMeasures) {
      double cellError = measureEntry.second;
      localMax = max(localMax,cellError);
    }
    _mesh->Comm()->MaxAll(&localMax, &globalMax, 1);
  }
};


int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  //////////////////////////////////////////////////////////////////////
  ///////////////////////  COMMAND LINE PARAMETERS  ////////////////////
  //////////////////////////////////////////////////////////////////////
  string problemChoice = "LidDriven";
  int spaceDim = 2;
  double rho = 1;
  double lambda = 1;
  double muS = 1; // solvent viscosity
  double muP = 1; // polymeric viscosity
  double alpha = 0;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  int numXElems = 2;
  int numYElems = 2;
  bool stokesOnly = false;
  bool enforceLocalConservation = false;
  bool useConformingTraces = true;
  bool evaluateJumps = false;
  string solverChoice = "KLU";
  string multigridStrategyString = "V-cycle";
  bool useCondensedSolve = false;
  bool useConjugateGradient = true;
  bool logFineOperator = false;
  double solverTolerance = 1e-8;
  int maxNonlinearIterations = 25;
  double nonlinearTolerance = 1e-8;
  // double minNonlinearTolerance = 10*solverTolerance;
  // double minNonlinearTolerance = 5e-7;
  double minNonlinearTolerance = 1e-4;
  // double minNonlinearTolerance = 4e-5;
  int maxLinearIterations = 1000;
  // bool computeL2Error = false;
  bool exportSolution = false;
  string norm = "Graph";
  string errorIndicator = "Energy";
  string outputDir = ".";
  string tag="";
  cmdp.setOption("problem", &problemChoice, "LidDriven, HalfHemker, Hemker");
  cmdp.setOption("rho", &rho, "rho");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("muS", &muS, "muS");
  cmdp.setOption("muP", &muP, "muP");
  cmdp.setOption("alpha", &alpha, "alpha");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("numYElems",&numYElems,"number of elements in y direction");
  cmdp.setOption("stokesOnly", "NavierStokesOnly", &stokesOnly, "couple only with Stokes, not Navier-Stokes");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy,CylinderBoundary,GoalOrientedDragCoeff");
  cmdp.setOption("enforceLocalConservation", "noLocalConservation", &enforceLocalConservation, "enforce local conservation principles at the element level");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("evaluateJumps", "DONOTevaluateJumps", &evaluateJumps, "evaluate the jump terms in the DPG* error estimator");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS, Pardiso");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  // cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);

  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr n    = Function::normal();

  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////

  ///////////////////////  SET PROBLEM PARAMETERS  /////////////////////
  Teuchos::ParameterList parameters;
  parameters.set("spaceDim", spaceDim);
  parameters.set("spatialPolyOrder", k);
  parameters.set("delta_k", delta_k);
  parameters.set("norm", norm);
  parameters.set("enforceLocalConservation",enforceLocalConservation);
  parameters.set("useConformingTraces", useConformingTraces);
  parameters.set("stokesOnly",stokesOnly);
  parameters.set("useConservationFormulation",false);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("rho", rho);
  parameters.set("lambda", lambda);
  parameters.set("muS", muS);
  parameters.set("muP", muP);
  parameters.set("alpha", alpha);


  ///////////////////////////  DECLARE MESH  ///////////////////////////

  MeshGeometryPtr spatialMeshGeom;
  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;


  double xLeft, xRight, height, yMax, cylinderRadius;
  if (problemChoice == "LidDriven")
  {
    // LID-DRIVEN CAVITY FLOW
    double x0 = 0.0, y0 = 0.0;
    double width = 1.0;
    height = 1.0;
    int horizontalCells = 2, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else if (problemChoice == "HalfHemker")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    xLeft = -15.0, xRight = 15.0;
    cylinderRadius = 1.0;
    yMax = 2.0*cylinderRadius;
    // MeshGeometryPtr spatialMeshGeom = MeshFactory::halfHemkerGeometry(xLeft,xRight,yMax,cylinderRadius);
    MeshGeometryPtr spatialMeshGeom = MeshFactory::halfConfinedCylinderGeometry(cylinderRadius);
    map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = spatialMeshGeom->edgeToCurveMap();
    globalEdgeToCurveMap = map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr >(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
    spatialMeshTopo = Teuchos::rcp( new MeshTopology(spatialMeshGeom) );
    spatialMeshTopo->setEdgeToCurveMap(globalEdgeToCurveMap, Teuchos::null);
  }
  else if (problemChoice == "Hemker")
  {
    // CONFINED CYLINDER BENCHMARK PROBLEM
    xLeft = -15.0, xRight = 15.0;
    cylinderRadius = 1.0;
    yMax = 2.0*cylinderRadius;
    spatialMeshGeom = MeshFactory::confinedCylinderGeometry(cylinderRadius);
    map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = spatialMeshGeom->edgeToCurveMap();
    globalEdgeToCurveMap = map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr >(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
    spatialMeshTopo = Teuchos::rcp( new MeshTopology(spatialMeshGeom) );
    spatialMeshTopo->setEdgeToCurveMap(globalEdgeToCurveMap, Teuchos::null);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  ///////////
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);
  OldroydBFormulationUW form(spatialMeshTopo, parameters);
  ///////////
  MeshPtr mesh = form.solutionIncrement()->mesh();
  if (globalEdgeToCurveMap.size() > 0)
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }


  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr solutionIncrement = form.solutionIncrement();
  SolutionPtr solutionBackground = form.solution();


  ///////////////////////////  DECLARE BC'S  ///////////////////////////
  BCPtr bc = form.solutionIncrement()->bc();
  VarPtr u1hat, u2hat, p;
  u1hat = form.u_hat(1);
  u2hat = form.u_hat(2);
  p     = form.p();

  if (problemChoice == "LidDriven")
  {
    // LID-DRIVEN CAVITY FLOW
    SpatialFilterPtr topBoundary = Teuchos::rcp( new TopLidBoundary );
    SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);

    //   top boundary:
    FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(1.0/64) );
    bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
    bc->addDirichlet(u2hat, topBoundary, zero);

    //   everywhere else:
    bc->addDirichlet(u1hat, otherBoundary, zero);
    bc->addDirichlet(u2hat, otherBoundary, zero);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else if (problemChoice == "HalfHemker")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
    SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(yMax);
    SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(0.0);
    SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));

    // inflow on left boundary
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(yMax) );
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 2, 2) );

    TFunctionPtr<double> u = Function::vectorize(u1_inflowFunction,u2_inflowFunction);

    form.addInflowCondition(leftBoundary, u);
    form.addInflowViscoelasticStress(leftBoundary, T11un_inflowFunction, T12un_inflowFunction, T22un_inflowFunction);

    // top+bottom
    form.addWallCondition(topBoundary);
    form.addSymmetryCondition(bottomBoundary);

    // outflow on right boundary
    // form.addOutflowCondition(rightBoundary, true); // true to impose zero traction by penalty (TODO)
    // form.addOutflowCondition(rightBoundary, yMax, muP, lambda, false); // false for zero flux variable
    form.addInflowCondition(rightBoundary, u);

    // no slip on cylinder
    // do this before symmetry condition in case of overlap of spatial filter
    form.addWallCondition(cylinderBoundary);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else if (problemChoice == "Hemker")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
    SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(yMax);
    SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(-yMax);
    SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));


    // UPDATE THIS FOR WHEN LAMBDA CHANGES

    // inflow on left boundary
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(yMax) );
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 2, 2) );

    TFunctionPtr<double> u = Function::vectorize(u1_inflowFunction,u2_inflowFunction);

    form.addInflowCondition(leftBoundary, u);
    form.addInflowViscoelasticStress(leftBoundary, T11un_inflowFunction, T12un_inflowFunction, T22un_inflowFunction);

    // top+bottom
    form.addWallCondition(topBoundary);
    form.addWallCondition(bottomBoundary);

    // outflow on right boundary
    // form.addOutflowCondition(rightBoundary, true); // true to impose zero traction by penalty (TODO)
    // form.addOutflowCondition(rightBoundary, yMax, muP, lambda, false); // false for zero flux variable
    form.addInflowCondition(rightBoundary, u);

    // no slip on cylinder
    form.addWallCondition(cylinderBoundary);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  //////////////////////////////////////////////////////////////////////
  ///////////////////////////////  SOLVE  //////////////////////////////
  //////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Teuchos::Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");

  if (commRank == 0)
    Solver::printAvailableSolversReport();
  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#if defined(HAVE_AMESOS_MUMPS) && defined(HAVE_MPI)
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
#ifdef HAVE_AMESOS_PARDISO_MKL
  solvers["Pardiso"] = Solver::getSolver(Solver::Pardiso, true);
#endif

  // choose local normal equation matrix calculation algorithm
  BFPtr bf = form.bf();
  // bf->setOptimalTestSolver(TBF<>::CHOLESKY);
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);

  ostringstream solnName;
  solnName << "GoalOrientedOldroydB" << "_" << norm << "_k" << k << "_dk" << delta_k << "_" << solverChoice;
  if (solverChoice[0] == 'G')
    solnName << "_" << multigridStrategyString;
  if (tag != "")
    solnName << "_" << tag;
  if (stokesOnly)
    solnName << "_Stokes";
  else
    solnName << "_NavierStokes";
  solnName << "_lambda_" << lambda;

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";

  ofstream dataFile(dataFileLocation);
  dataFile << "lambda\t "
           << "ref\t "
           << "elements\t "
           << "dofs\t "
           << "energy\t "
           << "solvetime\t "
           << "elapsed\t "
           << "iterations\t "
           << "drag coefficient (field)\t "
           << "drag coefficient (traction)\t "
           << "y-direction force\t "
           << "drag error estimate\t "
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter, functionExporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));
  string exporterName = "_DualSolutions";
  exporterName = solnName.str() + exporterName;
  functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));

  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {

    if (solverChoice != "SuperLUDist" && refIndex == 0)
      form.setSolver(solvers[solverChoice]);


    solverTime->start(true);

    ////////////////////////////////////////////////////////////////////
    //    Solve and accumulate solution
    ////////////////////////////////////////////////////////////////////
    int iterCount = 0;
    int iterationCount = 0;
    double l2Update = 1e10;
    double l2UpdateInitial = l2Update;
    while (l2Update > nonlinearTolerance*l2UpdateInitial && iterCount < maxNonlinearIterations && l2Update > minNonlinearTolerance)
    {      
      form.solveForIncrement();

      // Accumulate solution
      form.accumulate();

      ////////////////////////////////////////////////////////////////////
      // Compute L2 norm of update and increment counters
      ////////////////////////////////////////////////////////////////////
      l2Update = form.L2NormSolutionIncrement();

      if (commRank == 0)
        cout << "Nonlinear Update:\t " << l2Update << endl;

      if (iterCount == 0)
        l2UpdateInitial = l2Update;

      iterCount++;
    }

    double solveTime = solverTime->stop();

    double energyError = solutionIncrement->energyErrorTotal();

    // compute drag coefficient if Hemker problem
    double fieldDragCoefficient = 0.0;
    double fluxDragCoefficient = 0.0;
    double verticalForce = 0.0;
    double dragError = 0.0;
    if (problemChoice == "HalfHemker" || problemChoice == "Hemker")
    {
      SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius) );

      TFunctionPtr<double> boundaryRestriction = Function::meshBoundaryCharacteristic();

      // traction computed from field variables
      TFunctionPtr<double> n = TFunction<double>::normal();
      // L = muS*du/dx
      LinearTermPtr f_lt = - form.p()*n->x() + 2.0*form.L(1,1)*n->x() + form.T(1,1)*n->x()
                           + form.L(1,2)*n->y() + form.L(2,1)*n->y() + form.T(1,2)*n->y();
      TFunctionPtr<double> fieldTraction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, -f_lt ) );
      fieldTraction_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( fieldTraction_x*boundaryRestriction, cylinderBoundary) );
      fieldDragCoefficient = fieldTraction_x->integrate(mesh);

      // traction computed from flux
      TFunctionPtr<double> fluxTraction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(1)) );
      fluxTraction_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( fluxTraction_x*boundaryRestriction, cylinderBoundary) );
      fluxDragCoefficient = fluxTraction_x->integrate(mesh);

      // compute force in y-direction (from flux)
      TFunctionPtr<double> traction_y = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(2)) );
      traction_y = Teuchos::rcp( new SpatiallyFilteredFunction<double>( traction_y*boundaryRestriction, cylinderBoundary) );
      verticalForce = traction_y->integrate(mesh);

      dragError = ((fieldTraction_x-fluxTraction_x)*(fieldTraction_x-fluxTraction_x))->integrate(mesh);
      dragError = sqrt(2.0*M_PI)*sqrt(dragError);

      if (problemChoice == "HalfHemker")
      {
        fieldDragCoefficient = 2.0*fieldDragCoefficient;
        fluxDragCoefficient  = 2.0*fluxDragCoefficient;
        dragError = 2.0*dragError;
      }
    }

    if (commRank == 0)
    {
      cout << setprecision(8) 
        << "Lambda: " << lambda
        << " \nRefinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << " \tEnergy Error: " << energyError
        << " \nSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << " \tIteration Count: " << iterationCount
        << " \nDrag Coefficient (from field): " << fieldDragCoefficient
        << " \tDrag Coefficient (from traction): " << fluxDragCoefficient
        << " \ty-direction Force : " << verticalForce
        << " \tDrag error estimate : " << dragError
        << endl;
      dataFile << setprecision(8)
        << lambda
        << " " << refIndex
        << " " << mesh->numActiveElements()
        << " " << mesh->numGlobalDofs()
        << " " << energyError
        << " " << solveTime
        << " " << totalTimer->totalElapsedTime(true)
        << " " << iterationCount
        << " " << fieldDragCoefficient
        << " " << fluxDragCoefficient
        << " " << verticalForce
        << " " << dragError
        << endl;
    }

    if (exportSolution)
    {
      exporter->exportSolution(solutionBackground, refIndex);
      // exporter->exportSolution(solutionIncrement, refIndex);
      
      // output mesh with GnuPlotUtil
      ostringstream meshExportName;
      meshExportName << outputDir << "/" << solnName.str() << "/" << "ref" << refIndex << "_mesh";
      // int numPointsPerEdge = 3;
      bool labelCells = false;
      string meshColor = "black";
      GnuPlotUtil::writeComputationalMeshSkeleton(meshExportName.str(), mesh, labelCells, meshColor);
    }

    if (refIndex != numRefs)
    {
      ///////////////////  CHOOSE REFINEMENT STRATEGY  ////////////////////
      if (errorIndicator == "Energy")
      {
        form.refine();
      }
      else if (errorIndicator == "CylinderBoundary")
      {
        if (problemChoice == "HalfHemker" || problemChoice == "Hemker")
        {
          // SpatialFilterPtr cylinderBoundaryGenerous = Teuchos::rcp( new CylinderBoundaryGenerous(cylinderRadius));
          // ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryGenerous) );
          SpatialFilterPtr cylinderBoundaryExtreme = Teuchos::rcp( new CylinderBoundaryExtreme(cylinderRadius));
          ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryExtreme) );
          double energyThreshold = 0.2;
          RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, energyThreshold) );
          refStrategy->refine();
        }
        else
        {
          cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
          return Teuchos::null;
        }
      }
      else if (errorIndicator == "GoalOrientedDragCoeff")
      {
        if (problemChoice != "HalfHemker" && problemChoice != "Hemker")
        {
          cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
          return Teuchos::null;
        }

        // define goal functional
        SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius) );
        FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
        // FunctionPtr cylinderNormal_x, cylinderNormal_y;
        // cylinderNormal_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( n->x()*boundaryRestriction, cylinderBoundary) );
        // cylinderNormal_y = Teuchos::rcp( new SpatiallyFilteredFunction<double>( n->y()*boundaryRestriction, cylinderBoundary) );
        FunctionPtr v1_0 = Teuchos::rcp( new SpatiallyFilteredFunction<double>( 1.0 * boundaryRestriction, cylinderBoundary) );

        LinearTermPtr g_functional = v1_0 * form.sigman_hat(1);
        // g_functional = cylinderNormal_x * form.sigman_hat(1) + cylinderNormal_y * form.sigman_hat(2);

        // set second RHS
        solutionIncrement->setGoalOrientedRHS(g_functional);

        // solve again
        form.solveForIncrement();

        // construct DPG* solution
        bool excludeBoundaryTerms = false;
        const bool overrideMeshCheck = false; // testFunctional() default for third argument
        const int solutionOrdinal = 1; // solution corresponding to second RHS
        LinearTermPtr influence = form.bf()->testFunctional(solutionIncrement,excludeBoundaryTerms,overrideMeshCheck,solutionOrdinal);
        RieszRepPtr dualSoln = Teuchos::rcp(new RieszRep(mesh, form.bf()->graphNorm(), influence));
        dualSoln->computeRieszRep();

        FunctionPtr dualSoln_q, dualSoln_v1, dualSoln_v2, dualSoln_M1, dualSoln_M2, dualSoln_S11, dualSoln_S12, dualSoln_S22;
        dualSoln_q   =  Teuchos::rcp( new RepFunction<double>(form.q(), dualSoln) );
        dualSoln_v1  =  Teuchos::rcp( new RepFunction<double>(form.v(1), dualSoln) );
        dualSoln_v2  =  Teuchos::rcp( new RepFunction<double>(form.v(2), dualSoln) );
        dualSoln_M1  =  Teuchos::rcp( new RepFunction<double>(form.M(1), dualSoln) );
        dualSoln_M2  =  Teuchos::rcp( new RepFunction<double>(form.M(2), dualSoln) );
        dualSoln_S11 =  Teuchos::rcp( new RepFunction<double>(form.S(1,1), dualSoln) );
        dualSoln_S12 =  Teuchos::rcp( new RepFunction<double>(form.S(1,2), dualSoln) );
        dualSoln_S22 =  Teuchos::rcp( new RepFunction<double>(form.S(2,2), dualSoln) );

        // construct DPG* residual
        map<int, FunctionPtr > opDualSoln = bf()->applyAdjointOperatorDPGstar(dualSoln);
        FunctionPtr dualSolnResFxn = zero;
        for ( auto opDualSolnComponent  : opDualSoln  )
        {
          FunctionPtr f = opDualSolnComponent.second;
          dualSolnResFxn = dualSolnResFxn + f * f;
        }

        // evaluate all jump terms
        int cubatureDegreeEnrichment = delta_k;
        if (evaluateJumps == true)
        {
          // dualSoln_v1 = dualSoln_v1 + v1_0;
          bool weightBySideMeasure = false;
          std::map<GlobalIndexType, double> l2Jump_v1 = dualSoln_v1->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
          std::map<GlobalIndexType, double> l2Jump_v2 = dualSoln_v2->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);

          std::map<GlobalIndexType, double> l2Jump_total;
          const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
          double jump_v1=0.0;
          double jump_v2=0.0;
          for (auto cellID: myCellIDs)
          {
            l2Jump_total[cellID] = l2Jump_v1[cellID] + l2Jump_v2[cellID];
            if (weightBySideMeasure)
            {
              jump_v1 += l2Jump_v1[cellID];
              jump_v2 += l2Jump_v2[cellID];
            }
            else
            {
              double vol = mesh->getCellMeasure(cellID);
              double h = pow(vol, 1.0 / spaceDim);
              jump_v1 += pow(h, -1.0)*l2Jump_v1[cellID];
              jump_v2 += pow(h, -1.0)*l2Jump_v2[cellID];
            }
          }
          jump_v1 = sqrt(jump_v1);
          jump_v2 = sqrt(jump_v2);

          // print outputs
          double dualSolnRes = sqrt(dualSolnResFxn->l1norm(mesh, cubatureDegreeEnrichment));
          if (commRank == 0)
          {
            cout << setprecision(8) 
              << "\nDPG* residual: " << dualSolnRes
              << "\njump in v1:    " << jump_v1
              << "\njump in v2:    " << jump_v2
              << endl;
          }
        }

        // export DPG* solution
        vector<FunctionPtr> functionsToExport = {dualSoln_q, dualSoln_v1, dualSoln_v2, dualSoln_M1, dualSoln_M2, dualSoln_S11, dualSoln_S12, dualSoln_S22, dualSolnResFxn};
        vector<string> functionsToExportNames = {"dual_q", "dual_v1", "dual_v2", "dual_M1", "dual_M2", "dual_S11", "dual_S12", "dual_S22", "dualSolnResFxn"};

        if (exportSolution)
        {
          int numLinearPointsPlotting = max(k,15);
          functionExporter->exportFunction(functionsToExport, functionsToExportNames, refIndex, numLinearPointsPlotting);
        }

        // remove second RHS
        solutionIncrement->setGoalOrientedRHS(Teuchos::null);

        // refine
        form.solveForIncrement();
        form.accumulate();
        // form.refine();

        double refThreshold = 0.2;
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new GoalOrientedErrorIndicator<double>(solutionIncrement, dualSolnResFxn, cubatureDegreeEnrichment) );
        RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
        refStrategy->refine();

      }
    }
  }
  dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
