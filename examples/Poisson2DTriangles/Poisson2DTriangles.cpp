//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

#include "EpetraExt_RowMatrixOut.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int numElements = 2;
  vector<vector<double>> domainDim(3,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
  int polyOrder = 1, delta_k = 1;
  int spaceDim = 2;
  
  cmdp.setOption("numElements", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("x0", &domainDim[0][0] );
  cmdp.setOption("x1", &domainDim[0][1] );
  cmdp.setOption("y0", &domainDim[1][0] );
  cmdp.setOption("y1", &domainDim[1][1] );
  cmdp.setOption("z0", &domainDim[2][0] );
  cmdp.setOption("z1", &domainDim[2][1] );
  cmdp.setOption("spaceDim", &spaceDim);
  
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
  VarPtr phi = formCG.u();
  BFPtr bf = formCG.bf();
  
  int pToAddTest=0;
  double width = 1.0, height = 1.0;
  int horizontalElements = numElements, verticalElements = numElements;
  bool divideIntoTriangles=true;
  MeshPtr bubnovMesh = MeshFactory::quadMeshMinRule(bf, polyOrder, pToAddTest,
                                                    width, height,
                                                    horizontalElements, verticalElements,
                                                    divideIntoTriangles);
  
  cout << "numGlobalDofs = " << bubnovMesh->numGlobalDofs() << endl;
  cout << "numElements = " << bubnovMesh->numElements() << endl;
  cout << "numFieldDofs = " << bubnovMesh->numFieldDofs() << endl;
  
  cout << *(bubnovMesh->getElementType(0)->trialOrderPtr);
  
  bubnovMesh->getTopology()->printAllEntitiesInBaseMeshTopology();
  
  FunctionPtr x = Function::xn();
  
  RHSPtr rhs = RHS::rhs();
//  FunctionPtr f = Function::constant(1.0); // unit forcing
  FunctionPtr f = x * x;
  
  rhs->addTerm(f * q); // unit forcing
  
  IPPtr ip = Teuchos::null; // will give Bubnov-Galerkin
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(phi, SpatialFilter::allSpace(), Function::zero());
//  bc->addDirichlet(phi, SpatialFilter::allSpace(), x * x / 2.0);
  
  SolutionPtr solution = Solution::solution(bf, bubnovMesh, bc, rhs, ip);
  
  solution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
  solution->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
  
  solution->solve();
//  
//  HDF5Exporter exporter(bubnovMesh, "PoissonContinuousGalerkin");
//  exporter.exportSolution(solution);
  
  return 0;
}
