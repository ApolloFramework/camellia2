//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "Teuchos_CommandLineProcessor.hpp"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
    
  int rank = MPIWrapper::CommWorld()->MyPID();
  
  int spaceDim = 3;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim, conformingTraces);
  BFPtr bf = form.bf();
  
  FunctionPtr f = Function::constant(1.0);
  RHSPtr rhs = form.rhs(f);
  
  BCPtr bc = BC::bc();
  VarPtr u_hat = form.u_hat();
  SpatialFilterPtr everywhere = SpatialFilter::allSpace();
  bc->addDirichlet(u_hat, everywhere, Function::zero());
  
  vector<int> elementCounts = {1,1,1}; // x,y,z directions
  vector<double> domainDim = {1.0,1.0,1.0};
  int H1Order = 3;
  int delta_k = 3;
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, domainDim, elementCounts, H1Order, delta_k);
  
  // refine mesh uniformly 3 times -- result will be 8x8x8 mesh
  int numRefs = 3;
  for (int i=0; i<numRefs; i++)
  {
    mesh->hRefine(mesh->getActiveCellIDsGlobal());
  }
  
  SolutionPtr soln = Solution::solution(bf,mesh,bc,rhs,bf->graphNorm());
  soln->solve();
  
  HDF5Exporter exporter(mesh,"ch05-poisson-homogeneous");
  int numSubdivisions = 30; // coarse mesh -> more subdivisions
  int refNumber = 0;
  exporter.exportSolution(soln, refNumber, numSubdivisions);
  
  return 0;
}