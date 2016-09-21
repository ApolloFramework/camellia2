//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  PoissonFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#include "PoissonFormulation.h"
#include "RHS.h"

using namespace Camellia;

const string PoissonFormulation::S_U = "u";
const string PoissonFormulation::S_SIGMA = "\\sigma";

const string PoissonFormulation::S_U_HAT = "\\widehat{u}";
const string PoissonFormulation::S_SIGMA_N_HAT = "\\widehat{\\sigma}_n";

const string PoissonFormulation::S_V = "v";
const string PoissonFormulation::S_TAU = "\\tau";

PoissonFormulation::PoissonFormulation(int spaceDim, bool useConformingTraces, PoissonFormulationChoice formulationChoice)
{
  _spaceDim = spaceDim;

  if (formulationChoice == ULTRAWEAK)
  {
    Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
    Space u_hat_space = useConformingTraces ? HGRAD : L2;
    Space sigmaSpace = (spaceDim > 1) ? VECTOR_L2 : L2;

    // fields
    VarPtr u;
    VarPtr sigma;

    // traces
    VarPtr u_hat, sigma_n_hat;

    // tests
    VarPtr v;
    VarPtr tau;

    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar(S_U);
    sigma = vf->fieldVar(S_SIGMA, sigmaSpace);

    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    
    if (spaceDim > 1)
      u_hat = vf->traceVar(S_U_HAT, u, u_hat_space);
    else
      u_hat = vf->fluxVar(S_U_HAT, u * (Function::normal_1D() * parity), u_hat_space); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)

    TFunctionPtr<double> n = TFunction<double>::normal();

    if (spaceDim > 1)
      sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, sigma * (n * parity));
    else
      sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, sigma * (Function::normal_1D() * parity));

    v = vf->testVar(S_V, HGRAD);
    tau = vf->testVar(S_TAU, tauSpace);

    _poissonBF = Teuchos::rcp( new BF(vf) );

    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
      _poissonBF->addTerm(u, tau->dx());
      _poissonBF->addTerm(sigma, tau);
      _poissonBF->addTerm(-u_hat, tau);

      _poissonBF->addTerm(-sigma, v->dx());
      _poissonBF->addTerm(sigma_n_hat, v);
    }
    else
    {
      _poissonBF->addTerm(u, tau->div());
      _poissonBF->addTerm(sigma, tau);
      _poissonBF->addTerm(-u_hat, tau->dot_normal());

      _poissonBF->addTerm(-sigma, v->grad());
      _poissonBF->addTerm(sigma_n_hat, v);
    }
  }
  else if ((formulationChoice == PRIMAL) || (formulationChoice == CONTINUOUS_GALERKIN))
  {
    // field
    VarPtr u;
    
    // flux
    VarPtr sigma_n_hat;
    
    // tests
    VarPtr v;
    
    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar(S_U, HGRAD);
    
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();
    
    if (formulationChoice == PRIMAL)
    {
      if (spaceDim > 1)
        sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, u->grad() * (n * parity));
      else
        sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, u->dx() * (Function::normal_1D() * parity));
    }
    v = vf->testVar(S_V, HGRAD);
    
    _poissonBF = BF::bf(vf);
    _poissonBF->addTerm(-u->grad(), v->grad());

    if (formulationChoice == CONTINUOUS_GALERKIN)
    {
      FunctionPtr boundaryIndicator = Function::meshBoundaryCharacteristic();
      _poissonBF->addTerm(u->grad() * n, boundaryIndicator * v);
    }
    else // primal
    {
      _poissonBF->addTerm(sigma_n_hat, v);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported PoissonFormulationChoice");
  }
}

BFPtr PoissonFormulation::bf()
{
  return _poissonBF;
}

RHSPtr PoissonFormulation::rhs(FunctionPtr forcingFunction)
{
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(forcingFunction * v());
  return rhs;
}

// field variables:
VarPtr PoissonFormulation::u()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fieldVar(S_U);
}

VarPtr PoissonFormulation::sigma()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fieldVar(S_SIGMA);
}

// traces:
VarPtr PoissonFormulation::sigma_n_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fluxVar(S_SIGMA_N_HAT);
}

VarPtr PoissonFormulation::u_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->traceVar(S_U_HAT);
}

// test variables:
VarPtr PoissonFormulation::v()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_V, HGRAD);
}

VarPtr PoissonFormulation::tau()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  if (_spaceDim > 1)
    return vf->testVar(S_TAU, HDIV);
  else
    return vf->testVar(S_TAU, HGRAD);
}
