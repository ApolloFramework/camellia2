//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TestTemplate
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

// empty test file.  Copy (naming "MyClassTests.cpp", typically) and then add your tests below.

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"

using namespace Camellia;

namespace
{
  SolutionPtr kanschatStokesSolution(bool conformingTraces, int spaceDim, vector<int> elementCounts, int H1Order,
                                     bool enforceConservation)
  {
    double mu = 1.0;
    
    StokesVGPFormulation formulation = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    
    VarPtr p = formulation.p();
    
    BFPtr bf = formulation.bf();
    IPPtr graphNorm = bf->graphNorm();
    
    RHSPtr rhs = RHS::rhs();
    
    FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
    FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
    FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
    FunctionPtr exp_z = Teuchos::rcp( new Exp_z );
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    
    FunctionPtr u1_exact, u2_exact, u3_exact, p_exact;
    
    if (spaceDim == 2)
    {
      // this one was in the Cockburn Kanschat LDG Stokes paper
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y;
      p_exact = 2.0 * exp_x * sin_y;
    }
    else
    {
      // this one is inspired by the 2D one
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y + exp_z * y * cos_y;
      u3_exact = - exp_z * (cos_y - y * sin_y);
      p_exact = 2.0 * exp_x * sin_y + 2.0 * exp_z * cos_y;
    }
    
    // to ensure zero mean for p, need the domain carefully defined:
    vector<double> x0 = vector<double>(spaceDim,-1.0);
    
    BCPtr bc = BC::bc();
    
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(formulation.u_hat(1), boundary, u1_exact);
    bc->addDirichlet(formulation.u_hat(2), boundary, u2_exact);
    if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, u3_exact);
    
    FunctionPtr u_exact;
    if (spaceDim == 2)
    {
      u_exact = Function::vectorize({u1_exact,u2_exact});
    }
    else if (spaceDim == 3)
    {
      u_exact = Function::vectorize({u1_exact,u2_exact,u3_exact});
    }
    
    FunctionPtr forcingFunction = formulation.forcingFunction(u_exact, p_exact);
    
    rhs = formulation.rhs(forcingFunction);
    
    double width = 2.0;
    vector<double> dimensions(spaceDim,width);
    int delta_k = 2;
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0);
    
    bool useZeroMeanConstraints = true;
    if (!useZeroMeanConstraints)
    {
      vector<double> origin(spaceDim,0);
      IndexType vertexIndex;
      
      MeshTopologyViewPtr meshTopo = mesh->getTopology();
      bool foundVertex = meshTopo->getVertexIndex(origin, vertexIndex);
      foundVertex = MPIWrapper::globalOr(*mesh->Comm(), foundVertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!foundVertex, std::invalid_argument, "origin vertex not found on any rank");
      bc->addSpatialPointBC(p->ID(), 0, origin);
    }
    else
    {
      bc->addZeroMeanConstraint(p);
    }
    
    SolutionPtr soln = Solution::solution(bf, mesh, bc, rhs, graphNorm);
    
    if (enforceConservation)
    {
      LinearTermPtr uhat_dot_n;
      FunctionPtr n = Function::normal();
      for (int i=1; i<=spaceDim; i++)
      {
        VarPtr u_i = formulation.u_hat(i);
        if (uhat_dot_n == Teuchos::null)
        {
          uhat_dot_n = n->spatialComponent(i) * u_i;
        }
        else
        {
          uhat_dot_n->addTerm(n->spatialComponent(i) * u_i);
        }
      }
      soln->lagrangeConstraints()->addConstraint(uhat_dot_n == Function::zero());
    }
    
    return soln;
  }
  
  TEUCHOS_UNIT_TEST( Constraint, PoissonSimpleElementConstraint )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 1;
    int H1Order = 1, delta_k=1;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({1.0}, {3});
    
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, form.bf(), vector<int>{H1Order}, delta_k);
    BCPtr bc = BC::bc();
    RHSPtr rhs = RHS::rhs();
    SolutionPtr soln = Solution::solution(mesh,bc,rhs,form.bf()->graphNorm());
    
    // add a constraint that the trace "integral" on each element is 1.0
    VarPtr u_hat = form.u_hat();
    soln->lagrangeConstraints()->addConstraint(u_hat == Function::constant(1.0));
    
    int lagrangeOrdinal = 0;
    
    Epetra_Map partMap = soln->getPartitionMap();
    for (GlobalIndexType myCellID : mesh->cellIDsInPartition())
    {
      GlobalIndexType GID = soln->elementLagrangeIndex(myCellID, lagrangeOrdinal);
      int LID = partMap.LID(GID);
      // if GID is local (as it should be), LID should be non-negative
      TEST_COMPARE(LID, >=, 0);
    }
    
    soln->initializeStiffnessAndLoad();
    soln->populateStiffnessAndLoad();
    auto stiffness = soln->getStiffnessMatrix();
    auto load = soln->getRHSVector();
    
    Epetra_FECrsMatrix* feMatrix = dynamic_cast<Epetra_FECrsMatrix*>( stiffness.get() );
    feMatrix->GlobalAssemble();
    
    for (GlobalIndexType myCellID : mesh->cellIDsInPartition())
    {
      GlobalIndexType GID = soln->elementLagrangeIndex(myCellID, lagrangeOrdinal);
      int LID = partMap.LID(GID);
      
      // TODO: check values in stiffness and load, corresponding to the Constraint.
    }
  }
  
  TEUCHOS_UNIT_TEST( Constraint, StokesLocalConservation )
  {
    // just about any solution that is not exact will not by itself
    // be locally conservative.  So we use the Cockburn/Kanschat solution
    // to test our enforcement via element Lagrange constraints
    MPIWrapper::CommWorld()->Barrier();
    
    bool conformingTraces = true;
    int spaceDim = 2;
    vector<int> elementCounts = {2,2};
    int H1Order = 2;
    bool enforceConservation = true;
    SolutionPtr soln = kanschatStokesSolution(conformingTraces, spaceDim, elementCounts, H1Order, enforceConservation);
    
    bool exportMatrix = true;
    if (exportMatrix)
    {
      soln->setWriteMatrixToFile(true, "A.dat");
    }

    soln->solve();

    double mu = 1.0;
    StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
    FunctionPtr n = Function::normal();
    FunctionPtr u1hat_soln = Function::solution(u1_hat, soln);
    FunctionPtr u2hat_soln = Function::solution(u2_hat, soln);
    
    FunctionPtr uhat_soln = Function::vectorize(u1hat_soln, u2hat_soln);
    FunctionPtr flux = uhat_soln * n;
    
    double tol = 1e-12;
    auto myCellIDs = &soln->mesh()->cellIDsInPartition();
    for (GlobalIndexType cellID : *myCellIDs)
    {
      double cellFlux = flux->integrate(cellID, soln->mesh());
      if (abs(cellFlux) > tol)
      {
        success = false;
        out << "FAILURE: flux on cell " << cellID << " = " << cellFlux << endl;
      }
    }
    
//    HDF5Exporter exporter(soln->mesh(),"stokes-kanschat");
//    int numSubdivisions = 30; // coarse mesh -> more subdivisions
//    exporter.exportSolution(soln, 0, numSubdivisions);

  }
  
  
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace
