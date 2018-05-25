//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionsTests
//  Camellia
//
//  Created by Nate Roberts on 5/11/18.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "Functions.hpp"

using namespace Camellia;

namespace
{
  void testFunctionsMatch(FunctionPtr fExpected, FunctionPtr fActual, MeshPtr mesh, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    auto diff = fExpected - fActual;
    double err = diff->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
  }
  
  MeshPtr getMesh(int spaceDim, int polyOrder)
  {
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();
    
    IPPtr ip = bf->l2Norm();
    
    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    
    int H1Order = polyOrder + 1;
    // make a unit square mesh:
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    return mesh;
  }

  TEUCHOS_UNIT_TEST( Functions, IdentityMatrix )
  {
    double tol = 1e-14;
    int spaceDim = 2;
    int polyOrder = 1;
    auto mesh = getMesh(spaceDim, polyOrder);
    auto I = identityMatrix<double>(spaceDim);
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    auto v = Function::vectorize(one, zero);
    testFunctionsMatch(v, matvec(spaceDim, I, v), mesh, tol, out, success);
    testFunctionsMatch(2.0 * v, matvec(spaceDim, 2.0 * I, v), mesh, tol, out, success);
    v = Function::vectorize(zero, one);
    testFunctionsMatch(v, matvec(spaceDim, I, v), mesh, tol, out, success);
    testFunctionsMatch(2.0 * v, matvec(spaceDim, 2.0 * I, v), mesh, tol, out, success);
  }
  
//  TEUCHOS_UNIT_TEST( Functions, OuterProduct )
//  {
//    success = false;
//    out << "OuterProduct test not yet implemented";
//  }
} // namespace
