//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConvectionFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/6/14.
//
//

#include "ConvectionFormulation.h"

using namespace Camellia;

const string ConvectionFormulation::S_U = "u";
const string ConvectionFormulation::S_Q_N_HAT = "\\widehat{q}_n";
const string ConvectionFormulation::S_V = "v";

ConvectionFormulation::ConvectionFormulation(int spaceDim, TFunctionPtr<double> convectiveFunction)   // convectiveFunction should have zero divergence
{
  _spaceDim = spaceDim;

  if (_spaceDim != 2)
  {
    cout << "ConvectionFormulation only supports 2D right now.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ConvectionFormulation only supports 2D right now.");
  }

  VarFactoryPtr varFactory = VarFactory::varFactory();
  // traces:
  VarPtr qHat = varFactory->fluxVar(S_Q_N_HAT);

  // fields:
  VarPtr u = varFactory->fieldVar(S_U, L2);

  // test functions:
  VarPtr v = varFactory->testVar(S_V, HGRAD);

  _convectionBF = Teuchos::rcp( new BF(varFactory) );

  double theta = 1.0; // 0.5 for Crank-Nicolson (when using time-stepping)
//  _convectionBF->addTerm(u / dt, v); // transient term; to begin, ConvectionFormulation only supports steady state
  _convectionBF->addTerm(- theta * u, convectiveFunction * v->grad());
  _convectionBF->addTerm(qHat, v);
}

BFPtr ConvectionFormulation::bf()
{
  return _convectionBF;
}

// field variables:
VarPtr ConvectionFormulation::u()
{
  VarFactoryPtr vf = _convectionBF->varFactory();
  return vf->fieldVar(S_U);
}

// traces
VarPtr ConvectionFormulation::q_n_hat()
{
  VarFactoryPtr vf = _convectionBF->varFactory();
  return vf->fluxVar(S_Q_N_HAT);
}

// tests:
VarPtr ConvectionFormulation::v()
{
  VarFactoryPtr vf = _convectionBF->varFactory();
  return vf->testVar(S_V, HGRAD);
}
