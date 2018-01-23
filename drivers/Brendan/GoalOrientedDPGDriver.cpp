//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  PoissonGoalOrientedDPGDriver.cpp
//  Driver for goal-oriented adaptive mesh refinement for Poisson's BVP
//  Camellia
//
//  Created by Brendan Keith, December 2017.

#include "math.h"

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
#include "H1ProjectionFormulation.h"
#include "PoissonFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"

#include "CamelliaDebugUtility.h"


using namespace Camellia;

template <typename Scalar>
class DPGstarErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  int _cubatureDegreeEnrichment;
public:
  DPGstarErrorIndicator(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
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
      double dualresidual = sqrt(_dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment));
      _localErrorMeasures[cellID] = dualresidual;
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
  string formulation = "Poisson";
  string problemChoice = "SquareDomain";
  string formulationChoice = "ULTRAWEAK";
  int spaceDim = 2;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  string norm = "Graph";
  string errorIndicator = "Uniform";
  bool useConformingTraces = true;
  bool enrichTrial = false;
  string solverChoice = "KLU";
  bool exportSolution = false;
  string outputDir = ".";
  string tag="";
  cmdp.setOption("formulation", &formulation, "Poisson");
  cmdp.setOption("problem", &problemChoice, "SquareDomain, RectangularDomain");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("formulationChoice", &formulationChoice, "ULTRAWEAK");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variables");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy,Uniform,GoalOriented");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("enrichTrial", "classictrial", &enrichTrial, "use enriched u-variable");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (commRank == 0)
  {
    // for now, just print out the error indicator choice
    cout << "Selected options:\n";
    cout << " - Error Indicator: " << errorIndicator << endl;
    cout << endl << endl;
  }
  
  
  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);



  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one    = Function::constant(1);
  FunctionPtr zero   = Function::zero();
  FunctionPtr x      = Function::xn(1);
  FunctionPtr y      = Function::yn(1);
  FunctionPtr n      = Function::normal();
  // FunctionPtr heavi1 = Function::heaviside(1);
  const static double PI  = 3.141592653589793238462;
  FunctionPtr sin_pix = Teuchos::rcp( new Sin_ax(PI) );
  FunctionPtr sin_piy = Teuchos::rcp( new Sin_ay(PI) );
  FunctionPtr cos_pix = Teuchos::rcp( new Cos_ax(PI) );
  FunctionPtr cos_piy = Teuchos::rcp( new Cos_ay(PI) );
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );



  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////

  //////////////////////  DECLARE MESH TOPOLOGY  ///////////////////////
  // MeshGeometryPtr spatialMeshGeom;
  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;
  double width, height;
  if (problemChoice == "SquareDomain")
  {
    double x0 = 0.0, y0 = 0.0;
    width = 1.0;
    height = 1.0;
    int horizontalCells = 2, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else if (problemChoice == "RectangularDomain")
  {
    double x0 = 0.0, y0 = 0.0;
    width = 4.0;
    height = 1.0;
    int horizontalCells = 8, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  ///////////////////////  DECLARE BILINEAR FORM  //////////////////////
  PoissonFormulation form(spaceDim, useConformingTraces);

  BFPtr bf = form.bf();
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);


  //////////////////////  DECLARE TRIAL FUNCTIONS //////////////////////
  VarPtr u, sigma, u_hat, sigma_n_hat;
  u           = form.u();
  sigma       = form.sigma();
  u_hat       = form.u_hat(); 
  sigma_n_hat = form.sigma_n_hat();

  map<int, int> trialOrderEnhancements;
  if (enrichTrial)
    trialOrderEnhancements[u->ID()] = 1;


  //////////////////////  DECLARE TEST FUNCTIONS ///////////////////////
  VarPtr v, tau;
  v   = form.v();
  tau = form.tau();


  ///////////////////////////  DECLARE MESH  ///////////////////////////
  vector<int> H1Order = {k + 1};
  int testEnrichment = delta_k;
  MeshPtr mesh = Teuchos::rcp( new Mesh(spatialMeshTopo, bf, H1Order, testEnrichment, trialOrderEnhancements) ) ;
  if (globalEdgeToCurveMap.size() > 0) // only necessary if geometry is curvilinear
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }


  ///////////////////////  DECLARE INNER PRODUCT ///////////////////////
  map<string, IPPtr> poissonIPs;
  poissonIPs["Graph"] = bf->graphNorm();
  poissonIPs["Naive"] = bf->naiveNorm(spaceDim);
  IPPtr ip = poissonIPs[norm];


  //////////////////////  DECLARE EXACT SOLUTIONS  /////////////////////
  FunctionPtr u_exact, v_exact;
  FunctionPtr xx = x/width, yy = y/height;
  // u_exact = one;
  // u_exact = x * x + 2 * x * y;
  // u_exact = x * x * x * y + 2 * x * y * y;
  // u_exact = sin_pix * sin_piy;
  u_exact = xx * (1.0 - xx) * (xx/4.0 + (1.0 - 4.0*xx)*(1.0 - 4.0*xx) ) * yy * (1.0 - yy) * (yy/4.0 + (1.0 - 4.0*yy)*(1.0 - 4.0*yy) );

  xx = (width-x)/width;
  yy = (height-y)/height;
  // v_exact = zero;
  // v_exact = one;
  // v_exact = x * x + 2 * x * y;
  // v_exact = xx * (1.0 - xx) * (xx/4.0 + (1.0 - 4.0*xx)*(1.0 - 4.0*xx) ) * yy * (1.0 - yy) * (yy/4.0 + (1.0 - 4.0*yy)*(1.0 - 4.0*yy) );
  // v_exact = x * x * x * y + 2 * x * y * y;
  // v_exact = sin_pix * sin_piy;
  v_exact = cos_pix * cos_piy;


  ////////////////////////////  DECLARE RHS  ///////////////////////////
  FunctionPtr f;
  f = u_exact->dx()->dx() + u_exact->dy()->dy();
  RHSPtr rhs = form.rhs(f);


  ////////////////////  DECLARE BOUNDARY CONDITIONS ////////////////////
  BCPtr bc = BC::bc();
  if (problemChoice == "SquareDomain" || problemChoice == "RectangularDomain")
  {
    SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
    SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
    SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(width);
    SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(height);
    bc->addDirichlet(u_hat, x_equals_zero, u_exact);
    bc->addDirichlet(u_hat, y_equals_zero, u_exact);
    bc->addDirichlet(u_hat, x_equals_one, u_exact);
    bc->addDirichlet(u_hat, y_equals_one, u_exact);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);


  //////////////////////  DECLARE GOAL FUNCTIONAL //////////////////////
  LinearTermPtr g_functional;
  FunctionPtr g_u, g_sigma_x, g_sigma_y, g_sigma_hat;
  g_u = v_exact->dx()->dx() + v_exact->dy()->dy(); // field type
  g_functional = g_u * u;

  bool liftedBC = false;
  if (liftedBC)
  {
    g_sigma_x = v_exact->dx(); // LIFTED Dirichlet boundary data
    g_sigma_y = v_exact->dy(); // LIFTED Dirichlet boundary data
    bool overrideTypeCheck = true;
    g_functional->addTerm( - g_sigma_x * sigma->x(), overrideTypeCheck); // add flux type to field type
    g_functional->addTerm( - g_sigma_y * sigma->y(), overrideTypeCheck); // add flux type to field type
  }
  else
  {
    FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
    g_sigma_hat = v_exact * boundaryRestriction; // Dirichlet boundary data
    bool overrideTypeCheck = true;
    g_functional->addTerm( g_sigma_hat * sigma_n_hat, overrideTypeCheck); // add flux type to field type
  }

  // g_u = one - Function::heaviside(1);
  // g_functional = g_u * u;
  soln->setGoalOrientedRHS(g_functional);

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
#ifdef HAVE_AMESOS_MUMPS
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif

  ostringstream solnName;
  solnName << "GoalOrientedDPG" << "_" << norm << "_k" << k << "_" << solverChoice;
  if (tag != "")
    solnName << "_" << tag;
  solnName << "_" << formulation;

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";  

  ofstream dataFile(dataFileLocation);
  dataFile << "Ref\t "
           << "Elements\t "
           << "DOFs\t "
           << "Energy\t "
           << "Rate\t "
           << "Solvetime\t "
           << "Elapsed\t "
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  Teuchos::RCP<HDF5Exporter> functionExporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));
  string exporterName = "_DualSolutions";
  exporterName = solnName.str() + exporterName;
  functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));

  SolverPtr solver = solvers[solverChoice];
  double energyErrorPrvs, solnErrorPrvs, solnErrorL2Prvs, dualSolnErrorPrvs, dualSolnErrorL2Prvs, dualSolnResidualPrvs, outputErrorPrvs, numGlobalDofsPrvs;
  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
    solverTime->start(true);

    soln->solve(solver);

    double solveTime = solverTime->stop();
    double numGlobalDofs = mesh->numGlobalDofs();


    // compute error rep function / influence function
    bool excludeBoundaryTerms = false;
    const bool overrideMeshCheck = false; // testFunctional() default for third argument
    const int solutionOrdinal = 1; // solution corresponding to second RHS
    LinearTermPtr residual = rhs()->linearTerm() - bf->testFunctional(soln,excludeBoundaryTerms);
    LinearTermPtr influence = bf->testFunctional(soln,excludeBoundaryTerms,overrideMeshCheck,solutionOrdinal);
    RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
    RieszRepPtr dualSoln = Teuchos::rcp(new RieszRep(mesh, ip, influence));
    rieszResidual->computeRieszRep();
    dualSoln->computeRieszRep();

    // extract the test functions
    FunctionPtr psi_v, psi_tau, dualSoln_v, dualSoln_tau;
    psi_v =  Teuchos::rcp( new RepFunction<double>(form.v(), rieszResidual) );
    psi_tau =  Teuchos::rcp( new RepFunction<double>(form.tau(), rieszResidual) );
    dualSoln_v =  Teuchos::rcp( new RepFunction<double>(form.v(), dualSoln) );
    dualSoln_tau =  Teuchos::rcp( new RepFunction<double>(form.tau(), dualSoln) );

    vector<FunctionPtr> functionsToExport = {psi_v, psi_tau, dualSoln_v, dualSoln_tau};
    vector<string> functionsToExportNames = {"psi_v", "psi_tau", "dual_v", "dual_tau"};

    // compute error in DPG solution
    FunctionPtr soln_u = Function::solution(u, soln);
    FunctionPtr soln_sigma = Function::solution(sigma, soln);
    FunctionPtr e_u   = u_exact - soln_u;
    FunctionPtr e_sigma_x = u_exact->dx() - soln_sigma->x();
    FunctionPtr e_sigma_y = u_exact->dy() - soln_sigma->y();
    FunctionPtr solnErrorFunction = e_u*e_u + e_sigma_x*e_sigma_x + e_sigma_y*e_sigma_y;
    double solnError = sqrt(solnErrorFunction->l1norm(mesh));
    double solnErrorL2 = e_u->l2norm(mesh);
    double energyError = soln->energyErrorTotal();

    // compute error in DPG* solution
    FunctionPtr e_v   = v_exact - dualSoln_v;
    FunctionPtr e_v_dx   = v_exact->dx() - dualSoln_v->dx();
    FunctionPtr e_v_dy   = v_exact->dy() - dualSoln_v->dy();
    FunctionPtr e_tau_x = v_exact->dx() - dualSoln_tau->x();
    FunctionPtr e_tau_y = v_exact->dy() - dualSoln_tau->y();
    FunctionPtr e_tau_div = v_exact->dx()->dx() + v_exact->dy()->dy() - dualSoln_tau->div();
    FunctionPtr dualSolnErrorFunction = e_v*e_v + e_v_dx*e_v_dx + e_v_dy*e_v_dy + e_tau_x*e_tau_x + e_tau_y*e_tau_y + e_tau_div*e_tau_div;
    double dualSolnError = sqrt(dualSolnErrorFunction->l1norm(mesh));
    double dualSolnErrorL2 = e_v->l2norm(mesh);

    // compute residual in DPG* solution
    FunctionPtr res1 = dualSoln_tau->div() - g_u;
    FunctionPtr res2 = dualSoln_v->dx() - dualSoln_tau->x();  
    FunctionPtr res3 = dualSoln_v->dy() - dualSoln_tau->y();
    FunctionPtr dualSolnResidualFunction = res1*res1 + res2*res2 + res3*res3;

    int cubatureDegreeEnrichment = delta_k;
    double dualSolnResidual = sqrt(dualSolnResidualFunction->l1norm(mesh, cubatureDegreeEnrichment));

    // compute error in output (QOI)
    FunctionPtr outputError_function = g_u * (u_exact - soln_u);
    double outputError = abs(outputError_function->integrate(mesh, cubatureDegreeEnrichment));

    // compute rates
    double solnErrorRate = 0;
    double solnErrorL2Rate = 0;
    double energyRate = 0;
    double dualSolnErrorRate = 0;
    double dualSolnErrorL2Rate = 0;
    double dualSolnResidualRate = 0;
    double outputErrorRate = 0;
    if (refIndex != 0)
    {
      double denom = log(numGlobalDofsPrvs/numGlobalDofs);
      solnErrorRate =-spaceDim*log(solnErrorPrvs/solnError)/denom;
      solnErrorL2Rate =-spaceDim*log(solnErrorL2Prvs/solnErrorL2)/denom;
      energyRate =-spaceDim*log(energyErrorPrvs/energyError)/denom;
      dualSolnErrorRate =-spaceDim*log(dualSolnErrorPrvs/dualSolnError)/denom;
      dualSolnErrorL2Rate =-spaceDim*log(dualSolnErrorL2Prvs/dualSolnErrorL2)/denom;
      dualSolnResidualRate =-spaceDim*log(dualSolnResidualPrvs/dualSolnResidual)/denom;
      outputErrorRate =-spaceDim*log(outputErrorPrvs/outputError)/denom;
    }

    if (commRank == 0)
    {
      cout << setprecision(8) 
        << " \n\nRefinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << setprecision(4)
        << " \tSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << endl;
      cout << setprecision(4) 
        << " \nDPG Residual:        " << energyError
        << " \tRate: " << energyRate
        << " \nDPG Error (total):   " << solnError
        << " \tRate: " << solnErrorRate
        << " \nDPG Error (u-comp):  " << solnErrorL2
        << " \tRate: " << solnErrorL2Rate
        << endl;
      cout << setprecision(4) 
        << " \nDPG* Residual:       " << dualSolnResidual
        << " \tRate: " << dualSolnResidualRate
        << " \nDPG* Error: (total): " << dualSolnError
        << " \tRate: " << dualSolnErrorRate
        << " \nDPG* Error (v-comp): " << dualSolnErrorL2
        << " \tRate: " << dualSolnErrorL2Rate
        << endl;
      cout << setprecision(4) 
        << " \nGoal Error:          " << outputError
        << " \tRate: " << outputErrorRate
        << endl;
      dataFile << setprecision(8)
        << " " << refIndex
        << " " << mesh->numActiveElements()
        << " " << numGlobalDofs
        << setprecision(4)
        << " " << energyError
        << " " << energyRate
        << " " << solveTime
        << " " << totalTimer->totalElapsedTime(true)
        << " " << dualSolnError
        << " " << dualSolnErrorRate
        << endl;
    }

    solnErrorPrvs = solnError;
    solnErrorL2Prvs = solnErrorL2;
    energyErrorPrvs = energyError;
    dualSolnErrorPrvs = dualSolnError;
    dualSolnErrorL2Prvs = dualSolnErrorL2;
    dualSolnResidualPrvs = dualSolnResidual;
    outputErrorPrvs = outputError;
    numGlobalDofsPrvs = numGlobalDofs;

    if (exportSolution)
    {
      exporter->exportSolution(soln, refIndex);
      int numLinearPointsPlotting = max(k,15);
      functionExporter->exportFunction(functionsToExport, functionsToExportNames, refIndex, numLinearPointsPlotting);
    }

    if (refIndex != numRefs)
    {
      ///////////////////  CHOOSE REFINEMENT STRATEGY  ////////////////////
      double energyThreshold = 0.5;
      RefinementStrategyPtr refStrategy;
      if (errorIndicator == "Uniform")
      {
        energyThreshold = 0;
        refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
      }
      else if (errorIndicator == "Energy")
      {
        refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
      }
      else if (errorIndicator == "GoalOriented")
      {
        double refThreshold = 0.2;
        int cubatureDegreeEnrichment = delta_k;
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new GoalOrientedErrorIndicator<double>(soln, dualSolnResidualFunction, cubatureDegreeEnrichment) );
        refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
      }  
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized refinement strategy");
      }

      refStrategy->refine();
    } 

  }

  // dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "\n\nTotal time = " << totalTime << endl;

  return 0;
}
