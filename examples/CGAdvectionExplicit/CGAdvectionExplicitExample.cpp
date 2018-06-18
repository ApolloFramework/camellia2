//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "AbsFunction.h"
#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CubatureFactory.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "SpatiallyFilteredFunction.h"
#include "TrigFunctions.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#include "Epetra_Time.h"

#include "Intrepid_FunctionSpaceTools.hpp"

/*
 
 An experimental effort to use Camellia for an explicit CG Advection problem.
 
 The principal reason for this is to allow verification of code in Empire Fluid.
 
 */

using namespace Camellia;
using namespace Intrepid;
using namespace std;

const static double PI  = 3.141592653589793238462;

class U_Exact : public SimpleFunction<double>
{
  double value(double x)
  {
    double t = getTime(); // we need to make sure to update time during each time step
    return cos(2.*PI*(x-t));
  }
  TFunctionPtr<double> dx()
  {
    return Teuchos::null;  // should be 2.*PI*sin(2.*PI*(x-t)), but so far we don't need this
  }
  std::string displayString()
  {
    return "cos(2*pi*(x-t))";
  }
};

// **************************** Draft implementation of ERK tableau **************************** //
class Tableau
{
  int _numStages;
  vector<double> _A;
  vector<double> _b, _c;
public:
  Tableau(vector<double> &A, vector<double> &b, vector<double> &c) : _A(A), _b(b), _c(c)
  {
    _numStages = _b.size();
    TEUCHOS_ASSERT(_b.size() == _c.size());
    TEUCHOS_ASSERT(_A.size() == _numStages * _numStages);
  }
  
  int numStages() const
  {
    return _numStages;
  }
  
  double A(int i, int j) const
  {
    return _A[i * _numStages + j];
  }
  
  double b(int i) const
  {
    return _b[i];
  }
  
  double c(int i) const
  {
    return _c[i];
  }
  
  static Teuchos::RCP<Tableau> SSPRK3()
  {
    int numStages = 3;
    vector<double> A(numStages*numStages,0.0);
    vector<double> b(numStages, 0.0);
    vector<double> c(numStages, 0.0);
    
    A[1*numStages + 0] = 1.00;
    A[2*numStages + 0] = 0.25;
    A[2*numStages + 1] = 0.25;
    
    b[0] = 1./6.;
    b[1] = 1./6.;
    b[2] = 2./3.;
    
    c[0] = 0.;
    c[1] = 1.;
    c[2] = 1./2.;
    
    return Teuchos::rcp( new Tableau(A,b,c) );
  }
};

class ERKStepper
{
  SolutionPtr _soln;
  SolverPtr _solver;
  Teuchos::RCP<Tableau> _tableau;
  
  Teuchos::RCP<Epetra_FEVector> _uVector;      // "live" LHS in Solution
  Teuchos::RCP<Epetra_FEVector> _unVector;     // previous time step Solution
  Teuchos::RCP<Epetra_MultiVector> _rhsVector; // "live" RHS in Solution
  
  vector<Teuchos::RCP<Epetra_MultiVector>> _rhsStages;
public:
  ERKStepper(SolutionPtr soln, SolverPtr solver, Teuchos::RCP<Tableau> tableau) : _soln(soln), _solver(solver), _tableau(tableau)
  {
    soln->initializeLHSVector();
    soln->initializeStiffnessAndLoad();
    soln->setProblem(solver);
    soln->applyDGJumpTerms();
    soln->populateStiffnessAndLoad();
    
    int numStages = tableau->numStages();
    
    _uVector  = soln->getLHSVector();
    _unVector = Teuchos::rcp( new Epetra_FEVector(_uVector->Map(), _uVector->NumVectors()) );
    
    _rhsVector = soln->getRHSVector();
    
    _rhsStages.resize(numStages);
    for (int i=0; i<numStages; i++)
    {
      _rhsStages[i] = Teuchos::rcp( new Epetra_FEVector(_rhsVector->Map(), _rhsVector->NumVectors()) );
    }
  }
  
  void takeStep()
  {
    int solveSuccess = _soln->solveWithPrepopulatedStiffnessAndLoad(_solver);
    // TODO: finish this method
  }
};



// ********************************************************************************************* //
int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  // we solve with a simple manufactured solution
  int polyOrder = 1;
  int pToAddTest = 0;  // Bubnov-Galerkin, not DPG
  int meshWidth = 400;
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  using namespace Teuchos;
  
  FunctionPtr u_0 = rcp(new Cos_ax(2.*PI));
  
  FunctionPtr u_exact = rcp(new U_Exact);
  u_exact->setTime(0.); // need to update this each time step
  
  double dt = .00001;
  int numTimeSteps = 4;
  double h = 2.0 / meshWidth;
  double beta_norm = 1.0;
  FunctionPtr beta_x = Function::constant(beta_norm);
  double tau_h = h / beta_norm; // constant for our mesh; makes setup simpler
  
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u",HGRAD);
  VarPtr v = vf->testVar("v",HGRAD);
  
  BFPtr massBF = BF::bf(vf);
  massBF->addTerm(u, v);
  
  double xLeft =  -1.0;
  double xRight = 1.0;
  
  bool usePeriodicBCs = true;
  
  MeshPtr mesh = MeshFactory::intervalMesh(massBF, xLeft, xRight, meshWidth, polyOrder, pToAddTest, usePeriodicBCs);
  
  SolutionPtr soln = Solution::solution(massBF, mesh);
  
  const int solutionOrdinal = 0; // standard solution (as opposed to influence function)
  map<int,FunctionPtr> initialSolution = {{u->ID(),u_0}};
  soln->projectOntoMesh(initialSolution, solutionOrdinal);
  
  FunctionPtr u_N = Function::solution(u, soln); // u at previous time step
  
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(beta_x * u_N * v->dx());
  rhs->addTerm(-tau_h * beta_x * u_N->dx() * beta_x * v->dx());
  soln->setRHS(rhs);
  
  // We're relying on periodic BCs, so we just setup an empty BC object
  BCPtr bc = BC::bc();
  soln->setBC(bc);

  auto solnVector = soln->getLHSVector();
  auto solnVectorMap = solnVector->Map();
  Teuchos::RCP<Epetra_FEVector> prevSolnVector = Teuchos::rcp( new Epetra_FEVector(solnVectorMap) );
  
  soln->setWriteMatrixToMatrixMarketFile(true, "/tmp/Camellia_A.dat");
  soln->setWriteRHSToMatrixMarketFile(true, "/tmp/Camellia_b.dat");
  
  HDF5Exporter solnExporter(mesh,"CGAdvection","/tmp");
  
  FunctionPtr u_err = Teuchos::rcp( new AbsFunction(u_N - u_exact) );
  solnExporter.exportFunction({u_N, u_exact, u_err}, {"u_soln", "u_analytic", "u_err"}, 0.0); // time 0 solution
  
  
  double err_L1 = u_err->l1norm(mesh);
  if (rank==0) cout << "For timestep " << 0 << ", L^1 error: " << err_L1 << endl;
  
  // a very simple, crude time step scheme (forward Euler?)
  for (int N=1; N<numTimeSteps; N++)
  {
    // copy before solve:
    prevSolnVector->Update(1.0, *(soln->getLHSVector()), 0.0);
    
    soln->solve();
    // forward Euler sum:
    solnVector->Update(1.0, *prevSolnVector, dt);
    // distribute values back to local coefficients
    soln->importSolution();
    
    double t = dt*N;
    solnExporter.exportFunction({u_N, u_exact, u_err}, {"u_soln", "u_analytic", "u_err"}, t);
//    solnExporter.exportSolution(soln, t);
    
    u_exact->setTime(t);
    
    double err_L2 = u_err->l2norm(mesh);
    if (rank==0) cout << "For timestep " << N << ", L^2 error: " << err_L2 << endl;
  }
  
//  for (int N=1; N<numTimeSteps; N++)
//  {
//    // note that "solve()" will recompute the mass matrix at each time step, which is obviously inefficient
//    // it would be good to add some facility to Solution to allow reuse.
//    soln->solve();
//
//    // TODO: here, fill in the time-stepper
//
//    // update time for exact solution:
//    u_exact->setTime(N*dt);
//    // TODO: measure/report error here
//
//    FunctionPtr u_err = u_N - u_exact;
//    double err_L1 = u_err->l1norm(mesh);
//    if (rank==0) cout << "For timestep " << N << ", L^1 error: " << err_L1 << endl;
//  }

  return 0;
}
