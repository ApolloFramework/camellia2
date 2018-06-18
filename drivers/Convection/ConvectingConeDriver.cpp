#include "GnuPlotUtil.h"
#include "MeshFactory.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "SimpleFunction.h"
#include "Solver.h"
#include "TypeDefs.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include <Teuchos_GlobalMPISession.hpp>

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#define USE_HDF5
#else
#undef USE_HDF5
#endif
#undef USE_HDF5 // just because I have a version incompatibility thing happening right now...

#ifdef USE_HDF5
#include "HDF5Exporter.h"
#endif

using namespace Camellia;

class Cone_U0 : public SimpleFunction<double>
{
  double _r; // cone radius
  double _h; // height
  double _x0, _y0; // center
  bool _usePeriodicData; // if true, for x > 0.5 we set x = x-1; similarly for y
public:
  Cone_U0(double x0 = 0, double y0 = 0.25, double r = 0.1, double h = 1.0, bool usePeriodicData = true)
  {
    _x0 = x0;
    _y0 = y0;
    _r = r;
    _h = h;
    _usePeriodicData = usePeriodicData;
  }
  double value(double x, double y)
  {
    if (_usePeriodicData)
    {
      if (x > 0.5)
      {
        x = x - 1;
      }
      if (y > 0.5) y = y - 1;
    }
    double d = sqrt( (x-_x0) * (x-_x0) + (y-_y0) * (y-_y0) );
    double u = std::max(0.0, _h * (1 - d/_r));

    return u;
  }
};

class InflowFilterForClockwisePlanarRotation : public SpatialFilter
{
  double _xLeft, _yBottom, _xRight, _yTop;
  double _xMiddle, _yMiddle;
public:
  InflowFilterForClockwisePlanarRotation(double leftBoundary_x, double rightBoundary_x,
                                         double bottomBoundary_y, double topBoundary_y,
                                         double rotationCenter_x, double rotationCenter_y)
  {
    _xLeft = leftBoundary_x;
    _yBottom = bottomBoundary_y;
    _xRight = rightBoundary_x;
    _yTop = topBoundary_y;
    _xMiddle = rotationCenter_x;
    _yMiddle = rotationCenter_y;
  }
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool inflow;
    if (abs(x-_xLeft)<tol)
    {
      inflow = (y > _yMiddle);
    }
    else if (abs(x-_xRight)<tol)
    {
      inflow = (y < _yMiddle);
    }
    else if (abs(y-_yBottom)<tol)
    {
      inflow = (x < _xMiddle);
    }
    else if (abs(y-_yTop)<tol)
    {
      inflow = (x > _xMiddle);
    }
    else
    {
      inflow = false; // not a boundary point at all...
    }
    return inflow;
  }
};

int main(int argc, char *argv[])
{
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  bool useCondensedSolve = true;

  int numGridPoints = 64; // in x,y -- idea is to keep the overall order of approximation constant
  int k = 4; // poly order for u
  double theta = 0.5;
  int numTimeSteps = 2000;
  int numCells = -1; // in x, y (-1 so we can set a default if unset from the command line.)
  int numFrames = 50;
  int delta_k = 2;   // test space enrichment: should be 2 for 2D

  bool usePeriodicBCs = false;
  bool useConstantConvection = false;

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("numCells",&numCells,"number of cells in x and y directions");
  cmdp.setOption("theta",&theta,"theta weight for time-stepping");
  cmdp.setOption("numTimeSteps",&numTimeSteps,"number of time steps");
  cmdp.setOption("numFrames",&numFrames,"number of frames for export");

  cmdp.setOption("usePeriodicBCs", "useDirichletBCs", &usePeriodicBCs);
  cmdp.setOption("useConstantConvection", "useVariableConvection", &useConstantConvection);

  cmdp.setOption("useCondensedSolve", "useUncondensedSolve", &useCondensedSolve, "use static condensation to reduce the size of the global solve");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (numCells==-1) numCells = numGridPoints / k;

  if (rank==0)
  {
    cout << "solving on " << numCells << " x " << numCells << " mesh " << "of order " << k << ".\n";
  }

  set<int> timeStepsToExport;
  timeStepsToExport.insert(numTimeSteps);

  int timeStepsPerFrame = numTimeSteps / (numFrames - 1);
  if (timeStepsPerFrame==0) timeStepsPerFrame = 1;
  for (int n=0; n<numTimeSteps; n += timeStepsPerFrame)
  {
    timeStepsToExport.insert(n);
  }

  int H1Order = k + 1;

  const static double PI  = 3.141592653589793238462;

  double dt = 2 * PI / numTimeSteps;

  VarFactoryPtr varFactory = VarFactory::varFactory();
  // traces:
  VarPtr qHat = varFactory->fluxVar("\\widehat{q}");

  // fields:
  VarPtr u = varFactory->fieldVar("u", L2);

  // test functions:
  VarPtr v = varFactory->testVar("v", HGRAD);

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);

  FunctionPtr c;
  if (useConstantConvection)
  {
    c = Function::vectorize(Function::constant(0.5), Function::constant(0.5));
  }
  else
  {
    c = Function::vectorize(y-0.5, 0.5-x);
  }
//  FunctionPtr c = Function::vectorize(y, x);
  FunctionPtr n = Function::normal();

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  bf->addTerm(u / dt, v);
  bf->addTerm(- theta * u, c * v->grad());
//  bf->addTerm(theta * u_hat, (c * n) * v);
  bf->addTerm(qHat, v);

  double width = 2.0, height = 2.0;
  int horizontalCells = numCells, verticalCells = numCells;
  double x0 = -0.5;
  double y0 = -0.5;

  if (usePeriodicBCs)
  {
    x0 = 0.0;
    y0 = 0.0;
    width = 1.0;
    height = 1.0;
  }

  BCPtr bc = BC::bc();

  SpatialFilterPtr inflowFilter  = Teuchos::rcp( new InflowFilterForClockwisePlanarRotation (x0,x0+width,y0,y0+height,0.5,0.5));

  vector< PeriodicBCPtr > periodicBCs;
  if (! usePeriodicBCs)
  {
    //  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
    bc->addDirichlet(qHat, inflowFilter, Function::zero()); // zero BCs enforced at the inflow boundary.
  }
  else
  {
    periodicBCs.push_back(PeriodicBC::xIdentification(x0, x0+width));
    periodicBCs.push_back(PeriodicBC::yIdentification(y0, y0+height));
  }

  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, width, height,
                 horizontalCells, verticalCells, false, x0, y0, periodicBCs);

  FunctionPtr u0 = Teuchos::rcp( new Cone_U0(0.0, 0.25, 0.1, 1.0, usePeriodicBCs) );

  RHSPtr initialRHS = RHS::rhs();
  initialRHS->addTerm(u0 / dt * v);
  initialRHS->addTerm((1-theta) * u0 * c * v->grad());

  IPPtr ip;
//  ip = Teuchos::rcp( new IP );
//  ip->addTerm(v);
//  ip->addTerm(c * v->grad());
  ip = bf->graphNorm();

  // create two Solution objects; we'll switch between these for time steps
  SolutionPtr soln0 = Solution::solution(mesh, bc, initialRHS, ip);
  soln0->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln0 = Function::solution(u, soln0);
  FunctionPtr qHat_soln0 = Function::solution(qHat, soln0, false);

  RHSPtr rhs1 = RHS::rhs();
  rhs1->addTerm(u_soln0 / dt * v);
  rhs1->addTerm((1-theta) * u_soln0 * c * v->grad());

  SolutionPtr soln1 = Solution::solution(mesh, bc, rhs1, ip);
  soln1->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln1 = Function::solution(u, soln1);
  FunctionPtr qHat_soln1 = Function::solution(qHat, soln1, false);

  RHSPtr rhs2 = RHS::rhs(); // after the first solve on soln0, we'll swap out initialRHS for rhs2
  rhs2->addTerm(u_soln1 / dt * v);
  rhs2->addTerm((1-theta) * u_soln1 * c * v->grad());

  Teuchos::RCP<Solver> solver = Solver::getDirectSolver();

//  double energyErrorSum = 0;

  ostringstream filePrefix;
  filePrefix << "convectingCone_k" << k << "_t";
  int frameNumber = 0;

#ifdef USE_HDF5
  ostringstream dir_name;
  dir_name << "convectingCone_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());
#endif

  if (timeStepsToExport.find(0) != timeStepsToExport.end())
  {
    map<int,FunctionPtr> solnMap;
    solnMap[u->ID()] = u0; // project field variables
    if (rank==0) cout << "About to project initial solution onto mesh.\n";
    int solnOrdinal = 0;
    soln0->projectOntoMesh(solnMap, solnOrdinal);
    if (rank==0) cout << "...projected initial solution onto mesh.\n";
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
    if (rank==0) cout << "About to export initial solution.\n";
#ifdef USE_HDF5
    exporter.exportSolution(soln0,0);
#endif
  }

  if (rank==0) cout << "About to solve initial time step.\n";
  // first time step:
  soln0->setReportTimingResults(true); // added to gain insight into why MPI blocks in some cases on the server...
  if (useCondensedSolve) soln0->condensedSolve(solver);
  else soln0->solve(solver);
  soln0->setReportTimingResults(false);
//  energyErrorSum += soln0->energyErrorTotal();
  soln0->setRHS(rhs2);
  if (rank==0) cout << "Solved initial time step.\n";

  if (timeStepsToExport.find(1) != timeStepsToExport.end())
  {
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
#ifdef USE_HDF5
    exporter.exportSolution(soln0);
    if (saveSolutionFiles)
    {
      if (rank==0)
      {
//        filename << ".soln";
        soln0->save(filename.str());
        cout << endl << "wrote " << filename.str() << endl;
      }
    }
#endif
  }

  bool reportTimings = false;

  for (int n=1; n<numTimeSteps; n++)
  {
    bool odd = (n%2)==1;
    SolutionPtr soln_n = odd ? soln1 : soln0;
    if (useCondensedSolve) soln_n->solve(solver);
    else soln_n->solve(solver);
    if (reportTimings)
    {
      if (rank==0) cout << "time step " << n << ", timing report:\n";
      soln_n->reportTimings();
    }
    if (rank==0)
    {
      cout << "\x1B[2K"; // Erase the entire current line.
      cout << "\x1B[0E"; // Move to the beginning of the current line.
      cout << "Solved time step: " << n;
      flush(cout);
    }
    if (timeStepsToExport.find(n+1)!=timeStepsToExport.end())
    {
      ostringstream filename;
      filename << filePrefix.str() << frameNumber++;
#ifdef USE_HDF5
      double t = n * dt;
      if (odd)
      {
        exporter.exportSolution(soln1, t);
      }
      else
      {
        exporter.exportSolution(soln0, t);
      }
#endif
    }
//    energyErrorSum += soln_n->energyErrorTotal();
  }

//  if (rank==0) cout << "energy error, sum over all time steps: " << energyErrorSum << endl;

  return 0;
}
