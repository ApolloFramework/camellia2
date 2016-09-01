//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RieszRepTests
//  Camellia
//
//  Created by Nate Roberts on 2/16/15.
//
//

#include "RieszRep.h"
#include "PoissonFormulation.h"
#include "MeshFactory.h"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
TEUCHOS_UNIT_TEST( RieszRep, L2Norm )
{
  int spaceDim = 2;
  bool conformingTraces = true;

  PoissonFormulation form(spaceDim,conformingTraces);
  BFPtr bf = form.bf();

  IPPtr ip = bf->l2Norm();

  FunctionPtr weight = Function::xn(1);
  LinearTermPtr lt = weight * form.q();
  // Riesz rep should be simply (weight * q) (since the ip is (q,q) ).

  int H1Order = 1;
  // make a unit square mesh:
  vector<double> dimensions(2,1.0);
  vector<int> elementCounts(2,1);
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);

  RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, lt) );

  rieszRep->computeRieszRep();

  FunctionPtr repFxn = RieszRep::repFunction(form.q(), rieszRep);

  FunctionPtr expectedRepFxn = weight;

  double err = (repFxn - expectedRepFxn)->l2norm(mesh);

  double tol = 1e-14;
  TEST_COMPARE(err,<,tol);

  double expectedNorm = weight->l2norm(mesh);
  double actualNorm = rieszRep->getNorm();

  TEST_FLOATING_EQUALITY(expectedNorm,actualNorm, tol);
}
} // namespace
