//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int numElements = 20;
  vector<vector<double>> domainDim(2,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
  int polyOrder = 1;
  int spaceDim = 2;
  
  cmdp.setOption("meshWidth", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("x0", &domainDim[0][0] );
  cmdp.setOption("x1", &domainDim[0][1] );
  cmdp.setOption("y0", &domainDim[1][0] );
  cmdp.setOption("y1", &domainDim[1][1] );
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  vector<double> x0(spaceDim);
  vector<double> domainSize(spaceDim);
  vector<int> elementCounts(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    x0[d] = domainDim[d][0];
    domainSize[d] = domainDim[d][1] - x0[d];
    elementCounts[d] = numElements;
  }
  
  bool conformingTraces = true; // no difference for primal/continuous formulations
  PoissonFormulation formCG(spaceDim, conformingTraces, PoissonFormulation::CONTINUOUS_GALERKIN);
  VarPtr q = formCG.v();
  VarPtr u = formCG.u();
  BFPtr bf = formCG.bf();
  
  int pToAddTest=0;
  double width = 1.0, height = 1.0;
  int horizontalElements = numElements, verticalElements = numElements;
  bool divideIntoTriangles=true;
  MeshPtr bubnovMesh = MeshFactory::quadMeshMinRule(bf, polyOrder, pToAddTest,
                                                    width, height,
                                                    horizontalElements, verticalElements,
                                                    divideIntoTriangles);
  
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Function::constant(1.0); // unit forcing
  
  rhs->addTerm(f * q); // unit forcing
  
  IPPtr ip = Teuchos::null; // will give Bubnov-Galerkin
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u, SpatialFilter::allSpace(), Function::zero());
  
  SolutionPtr solution = Solution::solution(bf, bubnovMesh, bc, rhs, ip);
  
  Epetra_Time timer(*bubnovMesh->Comm());
  solution->solve();
  double solveTime = timer.ElapsedTime();
  
  if (rank == 0)
    cout << "Solved in " << solveTime << " seconds.\n";
  
  ostringstream vizPath;
  vizPath << "Poisson_Solution_" << spaceDim << "D_";
  vizPath << "Galerkin";
  vizPath << "_p" << polyOrder;
  HDF5Exporter vizExporter(bubnovMesh, vizPath.str(), ".");
  vizExporter.exportSolution(solution);
  
  if (rank == 0)
  {
    cout << "Exported solution to " << vizPath.str() << endl;
  }
  
  return 0;
}
