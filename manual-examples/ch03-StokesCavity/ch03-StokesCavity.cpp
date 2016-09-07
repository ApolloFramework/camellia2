//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

using namespace Camellia;

int main(int argc, char *argv[])
{
  VarFactoryPtr vf = VarFactory::varFactory();
  
  // field variables:
  VarPtr u = vf->fieldVar("u", VECTOR_L2);
  VarPtr p = vf->fieldVar("p", L2);
  VarPtr sigma1 = vf->fieldVar("sigma_1", VECTOR_L2);
  VarPtr sigma2 = vf->fieldVar("sigma_2", VECTOR_L2);
  
  // trace and flux variables:
  VarPtr u1_hat = vf->traceVar("u1_hat", HGRAD);
  VarPtr u2_hat = vf->traceVar("u2_hat", HGRAD);
  VarPtr tn1_hat = vf->fluxVar("tn1_hat", L2);
  VarPtr tn2_hat = vf->fluxVar("tn2_hat", L2);
  
  // test variables:
  VarPtr v1 = vf->testVar("v1", HGRAD);
  VarPtr v2 = vf->testVar("v2", HGRAD);
  VarPtr q = vf->testVar("q", HGRAD);
  VarPtr tau1 = vf->testVar("tau1", HDIV);
  VarPtr tau2 = vf->testVar("tau2", HDIV);
  
  // create BF object:
  BFPtr bf = BF::bf(vf);
  
  // get a normal function (will be useful in a moment):
  FunctionPtr n = Function::normal();
  
  double mu = 1.0;
  // add terms for v1:
  bf->addTerm(sigma1, v1->grad());
  bf->addTerm(-p, v1->dx());
  bf->addTerm(tn1_hat, v1);
  
  // add terms for v2:
  bf->addTerm(sigma2, v2->grad());
  bf->addTerm(-p, v2->dy());
  bf->addTerm(tn2_hat, v2);
  
  // add terms for q:
  bf->addTerm(-u,q->grad());
  bf->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  
  // add terms for tau1:
  bf->addTerm(mu * u->x(), tau1->div());
  bf->addTerm(sigma1, tau1);
  bf->addTerm(-u1_hat, tau1 * n);
  
  // add terms for tau2:
  bf->addTerm(mu * u->y(), tau2->div());
  bf->addTerm(sigma2, tau2);
  bf->addTerm(-u2_hat, tau2 * n);
  
  return 0;
}