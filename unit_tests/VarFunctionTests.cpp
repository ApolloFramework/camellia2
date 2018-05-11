//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VarFunctionTests
//  Camellia
//
//  Created by Nate Roberts on 5/7/18.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "VarFunction.h"

using namespace Camellia;

namespace
{
  void testTermsMatch(LinearTermPtr ltExpected, LinearTermPtr ltActual, MeshPtr mesh, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    LinearTermPtr diff = ltExpected - ltActual;
    
    BFPtr bf = mesh->bilinearForm();
    
    IPPtr ip = bf->l2Norm();
    
    RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, diff) );
    
    rieszRep->computeRieszRep();
    
    double err = rieszRep->getNorm();

    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( VarFunction, LinearJacobian )
  {
    // Just test that when we take a linear abstract function
    // and evaluate its Jacobian at an arbitrary solution, we get
    // the same thing that we get when we use the equivalent LinearTerms

    // basic form of the test will be something like:
//    LinearTermPtr lt;
//    VarFunction<double> abstractFunction = VarFunction<double>::abstractFunction(lt);
//
//    linearized = abstractFunction->jacobian();
//
//    assert that (linearized - lt) is everywhere zero.
    
    double tol = 1e-14;
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();
    
    IPPtr ip = bf->l2Norm();
    
    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    
    int H1Order = 2;
    // make a unit square mesh:
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    VarPtr u = form.u();
    
    auto uVarFunction = VarFunction<double>::abstractFunction(u);
    
    map<int, FunctionPtr> valueMap; // we can leave empty for this first test
    
    // jacobian of a linear term should be same as the term itself
    auto uLT = uVarFunction->jacobian(valueMap);
    
    auto uLTExpected = 1.0 * u;
    
    testTermsMatch(uLTExpected, uLT, mesh, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( VarFunction, ProductJacobian )
  {
    // Just test that when we take a product abstract function
    // and evaluate its Jacobian at an arbitrary solution, we get
    // the same thing that we get when we use the equivalent LinearTerms
    
    // basic form of the test will be something like:
    //    auto u = VarFunction<double>::abstractFunction(form.u());
    //    LinearTermPtr lt = 2.0 * u_fxn * u;
    //    VarFunction<double> abstractFunction = VarFunction<double>::abstractFunction(u*u);
    //
    //    linearized = abstractFunction->jacobian(u_fxn);
    //
    //    assert that (linearized - lt) is everywhere zero.
    
    double tol = 1e-14;
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();
    
    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    
    int H1Order = 2;
    // make a unit square mesh:
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    VarPtr u = form.u();
    
    auto uVarFunction = VarFunction<double>::abstractFunction(u);
    
    map<int, FunctionPtr> valueMap;
    auto x = Function::xn(1);
    valueMap[u->ID()] = x;
    
    auto uSquared = uVarFunction * uVarFunction;
    auto uSquaredJacobianLT = uSquared->jacobian(valueMap);
    
    auto uSquaredExpectedJacobian = 2.0 * valueMap[u->ID()] * u;
    
    testTermsMatch(uSquaredExpectedJacobian, uSquaredJacobianLT, mesh, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( VarFunction, QuotientJacobian )
  {
    // Just test that when we take a quotient abstract function
    // and evaluate its Jacobian at an arbitrary solution, we get
    // the same thing that we get when we use the equivalent LinearTerms
    
    // basic form of the test will be something like:
    //    auto u = VarFunction<double>::abstractFunction(form.u());
    //    VarFunction<double> abstractFunction = VarFunction<double>::abstractFunction(1.0 / u);
    //
    //    LinearTermPtr lt = -1.0 / (u_fxn * u_fxn) * u;
    //
    //    linearized = abstractFunction->jacobian(u_fxn);
    //
    //    assert that (linearized - lt) is everywhere zero.
    
    double tol = 1e-14;
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();
    
    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    
    int H1Order = 2;
    // make a unit square mesh:
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    VarPtr u = form.u();
    
    auto uVarFunction = VarFunction<double>::abstractFunction(u);
    
    map<int, FunctionPtr> valueMap;
    auto uFxn = Function::xn(1);
    valueMap[u->ID()] = uFxn;
    
    auto oneOverU = 1.0 / uVarFunction;
    auto oneOverUJacobian = oneOverU->jacobian(valueMap);
    
    auto oneOverUExpectedJacobian = - 1.0 / (uFxn * uFxn) * u;
    
    testTermsMatch(oneOverUExpectedJacobian, oneOverUJacobian, mesh, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( VarFunction, SumJacobian )
  {
    // Just test that when we take a sum abstract function
    // and evaluate its Jacobian at an arbitrary solution, we get
    // the same thing that we get when we use the equivalent LinearTerms
    
    // basic form of the test will be something like:
    //    auto u = VarFunction<double>::abstractFunction(form.u());
    //    LinearTermPtr lt = 2.0 * u;
    //    VarFunction<double> abstractFunction = VarFunction<double>::abstractFunction(u + u);
    //
    //    linearized = abstractFunction->jacobian(u_fxn);
    //
    //    assert that (linearized - lt) is everywhere zero.
    
    double tol = 1e-14;
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();
    
    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    
    int H1Order = 2;
    // make a unit square mesh:
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,1);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    VarPtr u = form.u();
    
    auto uVarFunction = VarFunction<double>::abstractFunction(u);
    
    map<int, FunctionPtr> valueMap; // we can leave empty for this test
    
    auto expectedLT = 2.0 * u;
    auto jacobianLT = (uVarFunction + uVarFunction)->jacobian(valueMap);
    testTermsMatch(expectedLT, jacobianLT, mesh, tol, out, success);
  }
} // namespace
