//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#ifdef HAVE_MOAB

#include "BC.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "Solution.h"
#include "Solver.h"
#include "TypeDefs.h"

#include "Epetra_Time.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  string meshFileName;
  int polyOrder = 1;
  std::string solverName = "KLU";
  
  cmdp.setOption("meshFile", &meshFileName, "Mesh to load from an HDF5 file using MOAB." );
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("solver", &solverName);
  
  if (rank == 0)
    Solver::printAvailableSolversReport();
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  bool readInParallel = false; // distributed read not currently supported
  MeshTopologyPtr meshTopo = MeshFactory::importMOABMesh(meshFileName, readInParallel);
  
  int spaceDim = meshTopo->getDimension();
  int cellCount = meshTopo->activeCellCount();
  if (rank==0) cout << spaceDim << "D mesh topology has " << cellCount << " cells.\n";
  
  // solve a homogeneous Poisson problem with unit forcing on this mesh
  bool conformingTraces = true;
  auto formulationChoice = PoissonFormulation::ULTRAWEAK;
  PoissonFormulation form(spaceDim, conformingTraces, formulationChoice);
  
  auto bf    = form.bf();
  auto rhs   = form.rhs(Function::constant(1.0));
  auto bc    = BC::bc();
  auto ip    = bf->graphNorm();
  
  auto u_hat = form.u_hat();
  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
  
  int H1Order = polyOrder + 1; // H^1 order is 1 higher than L^2 order
  int delta_p = 1; // governs fidelity of optimal test solve -- theory says spaceDim is good, but can get away with less in practice
  
  auto mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_p);
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  
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
  if (solvers.find(solverName) == solvers.end())
  {
    if (rank == 0)
    {
      cout << "Error: requested solver " << solverName << " not supported.\n";
      cout << "Available solvers: ";
      for (auto entry : solvers)
      {
        cout << entry.first << " ";
      }
      cout << endl;
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  auto solver = solvers[solverName];
  
  solution->setUseCondensedSolve(true);
  Epetra_Time timer(*mesh->Comm());
  solution->solve(solver);
  double solveTime = timer.ElapsedTime();
  
  if (rank == 0)
    cout << "Solved in " << solveTime << " seconds.\n";
  
  ostringstream vizPath;
  vizPath << "Poisson_Solution_" << spaceDim << "D_DPG_";
  vizPath << "p" << polyOrder;
  HDF5Exporter vizExporter(mesh, vizPath.str(), ".");
  vizExporter.exportSolution(solution);
  
  if (rank == 0)
  {
    cout << "Exported solution to " << vizPath.str() << endl;
  }
  
  return 0;
}
#else

int main(int argc, char *argv[])
{
  cout << "Error - HAVE_MOAB preprocessor macro not defined.\n";
  
  return 0;
}

#endif
