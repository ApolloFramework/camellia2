//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  ConvectionDiffusionReactionFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts Mar 3, 2016.
//
//

#include "ConvectionDiffusionReactionFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;

const string ConvectionDiffusionReactionFormulation::S_U = "u";
const string ConvectionDiffusionReactionFormulation::S_SIGMA = "sigma";

const string ConvectionDiffusionReactionFormulation::S_UHAT = "uhat";
const string ConvectionDiffusionReactionFormulation::S_SIGMA_N = "sigma_n";

const string ConvectionDiffusionReactionFormulation::S_V = "v";
const string ConvectionDiffusionReactionFormulation::S_TAU = "tau";

ConvectionDiffusionReactionFormulation::ConvectionDiffusionReactionFormulation(FormulationChoice formulation, int spaceDim,
                                                                               FunctionPtr beta, double epsilon, double alpha)
{
  _formulationChoice = formulation;
  _spaceDim = spaceDim;
  _beta = beta;
  _epsilon = epsilon;
  _alpha = alpha;

  Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
  Space uhat_space = HGRAD;
  Space sigmaSpace;

  // fields
  VarPtr u;
  VarPtr sigma;

  // traces
  VarPtr uhat, sigma_n; // sigma_n is the normal trace of the whole thing under the divergence operator, not just sigma

  // tests
  VarPtr v;
  VarPtr tau;

  _vf = VarFactory::varFactory();

  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> parity = TFunction<double>::sideParity();

  double sqrt_epsilon = sqrt(_epsilon);
  switch (_formulationChoice)
  {
    case ULTRAWEAK:
      sigmaSpace = (spaceDim > 1) ? VECTOR_L2 : L2;
      u = _vf->fieldVar(S_U);
      sigma = _vf->fieldVar(S_SIGMA, sigmaSpace);
      
      if (spaceDim > 1)
        uhat = _vf->traceVar(S_UHAT, u, uhat_space);
      else
        uhat = _vf->fluxVar(S_UHAT, u, uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
      
      if (spaceDim > 1)
        sigma_n = _vf->fluxVar(S_SIGMA_N, (sqrt_epsilon * sigma - _beta * u) * (n * parity));
      else
        sigma_n = _vf->fluxVar(S_SIGMA_N, sqrt_epsilon * sigma - _beta * u);
      
      v = _vf->testVar(S_V, HGRAD);
      tau = _vf->testVar(S_TAU, tauSpace);
      
      _bf = Teuchos::rcp( new BF(_vf) );
      
      if (spaceDim==1)
      {
        // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
        _bf->addTerm(sigma, tau);
        _bf->addTerm(sqrt_epsilon * u, tau->dx());
        _bf->addTerm(-sqrt_epsilon * uhat, tau);
        
        _bf->addTerm(sqrt_epsilon * sigma - _beta * u, v->dx());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      else
      {
        _bf->addTerm(sigma, tau);
        _bf->addTerm(sqrt_epsilon * u, tau->div());
        _bf->addTerm(-sqrt_epsilon * uhat, tau->dot_normal());
        
        _bf->addTerm(sqrt_epsilon * sigma -_beta * u, v->grad());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      break;
    case PRIMAL:
      u = _vf->fieldVar(S_U, HGRAD);
      if (spaceDim == 1)
        sigma_n = _vf->fluxVar(S_SIGMA_N, (_beta * u - _epsilon * u->dx()));
      else
        sigma_n = _vf->fluxVar(S_SIGMA_N, (_beta * u - _epsilon * u->grad()) * (n * parity));
      
      v = _vf->testVar(S_V, HGRAD);
      
      _bf = Teuchos::rcp( new BF(_vf) );
      
      if (spaceDim==1)
      {
        // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
        _bf->addTerm(_epsilon * u->dx() - _beta * u, v->dx());
        _bf->addTerm(- sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      else
      {
        _bf->addTerm(_epsilon * u->grad() - _beta * u, v->grad());
        _bf->addTerm(-sigma_n, v);
        _bf->addTerm(_alpha * u, v);
      }
      break;
    case SUPG:
      u = _vf->fieldVar(S_U, HGRAD);
      v = _vf->testVar(S_V, HGRAD);
    {
      // set up stabilization term (from Egger & Schöberl, via Broersen & Stevenson)
      //            max(h|b| - 2e, 0)/|b|^2
      // where h is element width, b is beta, and e is epsilon.
      // See http://www.asc.tuwien.ac.at/~schoeberl/wiki/publications/MixedHybridDG.pdf (Section 5)
      
      FunctionPtr h = Function::h();
      FunctionPtr beta_norm_squared = Function::zero();
      if (spaceDim == 1)
      {
        beta_norm_squared = beta * beta;
      }
      else
      {
        for (int comp = 1; comp <= spaceDim; comp++)
        {
          FunctionPtr beta_comp = beta->spatialComponent(comp);
          beta_norm_squared = beta_norm_squared + beta_comp * beta_comp;
        }
      }
      FunctionPtr beta_norm = Function::sqrtFunction(beta_norm_squared);
      FunctionPtr Egger_Schoberl = Function::max(h * beta_norm - 2 * _epsilon, Function::zero() / beta_norm_squared);
      // per Vijay, the stabilization weight should not be more than h / 2
      FunctionPtr stabilizationWeight = Function::min(Egger_Schoberl, h / 2.0);
      _stabilizationWeight = ParameterFunction::parameterFunction(stabilizationWeight);
    }
      _bf = Teuchos::rcp( new BF(_vf) );
      
      if (spaceDim==1)
      {
        _bf->addTerm(_epsilon * u->dx() - _beta * u, v->dx());
        _bf->addTerm(_alpha * u, v);
        
        // stabilization term
        FunctionPtr tau = _stabilizationWeight;
        _bf->addTerm( tau * -_epsilon * u->laplacian(), _beta * v->dx());
        _bf->addTerm( tau * _beta * u->dx(), _beta * v->dx());
        _bf->addTerm( tau * _alpha * u, _beta * v->dx());
      }
      else
      {
        _bf->addTerm(_epsilon * u->grad() - _beta * u, v->grad());
        _bf->addTerm(_alpha * u, v);
        
        // stabilization term
        FunctionPtr tau = _stabilizationWeight;
        _bf->addTerm( tau * -_epsilon * u->laplacian(), _beta * v->grad());
        _bf->addTerm( tau * _beta * u->grad(), _beta * v->grad());
        _bf->addTerm( tau * _alpha * u, _beta * v->grad());
      }
      break;
    case LEAST_SQUARES:
      sigmaSpace = (spaceDim > 1) ? HDIV : HGRAD;

      u     = _vf->fieldVar(S_U, HGRAD);
      sigma = _vf->fieldVar(S_SIGMA, sigmaSpace);
      v   = _vf->testVar(S_V, HGRAD);
      tau = _vf->testVar(S_TAU, tauSpace);

      _bf = Teuchos::rcp( new BF(_vf) );

      if (spaceDim==1)
      {
        _bf->addTerm( sigma - _epsilon * u->dx(), tau );
        _bf->addTerm( -_epsilon * sigma + _epsilon * _epsilon * u->dx(), v->dx() );

        _bf->addTerm( sigma->dx() - _beta * u->dx() - _alpha * u, tau->dx());
        _bf->addTerm( -sigma->dx() + _beta * u->dx() + _alpha * u, _beta * v->dx());
        _bf->addTerm( -_alpha * sigma->dx() + _alpha * _beta * u->dx() + _alpha * _alpha * u, v);
      }
      else
      {
        _bf->addTerm( sigma - _epsilon * u->grad(), tau );
        _bf->addTerm( -_epsilon * sigma + _epsilon * _epsilon * u->grad(), v->grad() );
        
        _bf->addTerm( sigma->div() - _beta * u->grad() - _alpha * u, tau->div());
        _bf->addTerm( -sigma->div() + _beta * u->grad() + _alpha * u, _beta * v->grad());
        _bf->addTerm( -_alpha * sigma->div() + _alpha * _beta * u->grad() + _alpha * _alpha * u, v);
      }
  }
}

FunctionPtr ConvectionDiffusionReactionFormulation::forcingFunction(FunctionPtr u_exact)
{
  // -epsilon \Delta u + div (beta * u) + alpha u
  if (_spaceDim > 1)
    return -_epsilon * u_exact->grad(_spaceDim)->div() + (_beta * u_exact)->div() + _alpha * u_exact;
  else
    return -_epsilon * u_exact->dx()->dx() + (_beta * u_exact)->dx() + _alpha * u_exact;
}

BFPtr ConvectionDiffusionReactionFormulation::bf()
{
  return _bf;
}

RHSPtr ConvectionDiffusionReactionFormulation::rhs(FunctionPtr f)
{
  RHSPtr rhs = RHS::rhs();
  if (_formulationChoice == LEAST_SQUARES)
  {
    if (_spaceDim == 1)
    {
      rhs->addTerm( -f * tau()->dx());
      rhs->addTerm( f * _beta * v()->dx());
      rhs->addTerm( f * _alpha * v());
    }
    else
    {
      rhs->addTerm( -f * tau()->div());
      rhs->addTerm( f * _beta * v()->grad());
      rhs->addTerm( f * _alpha * v());
    }
  }
  else
  {
    rhs->addTerm(f * v());
    if (_formulationChoice == SUPG)
    {
      // add stabilization term
      FunctionPtr tau = _stabilizationWeight;
      if (_spaceDim == 1)
        rhs->addTerm(tau * f * (_beta * v()->dx()));
      else
        rhs->addTerm(tau * f * (_beta * v()->grad()));
    }
  }
  return rhs;
}

LinearTermPtr ConvectionDiffusionReactionFormulation::residual(SolutionPtr soln)
{
  LinearTermPtr bfEval = _bf->testFunctional(soln);
  return bfEval - soln->rhs()->linearTerm();
}

void ConvectionDiffusionReactionFormulation::setSUPGStabilizationWeight(FunctionPtr stabilizationWeight)
{
  if (_formulationChoice != SUPG)
  {
    cout << "setSUPGStabilizationWeight() requires that formulation was constructed with SUPG formulation choice.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setSUPGStabilizationWeight() requires that formulation was constructed with SUPG formulation choice.");
  }
  _stabilizationWeight->setValue(stabilizationWeight);
}

VarPtr ConvectionDiffusionReactionFormulation::sigma()
{
  return _vf->fieldVar(S_SIGMA);
}

VarPtr ConvectionDiffusionReactionFormulation::sigma_n()
{
  return _vf->fluxVar(S_SIGMA_N);
}

FunctionPtr ConvectionDiffusionReactionFormulation::SUPGStabilizationWeight()
{
  return _stabilizationWeight;
}

VarPtr ConvectionDiffusionReactionFormulation::u()
{
  return _vf->fieldVar(S_U);
}

VarPtr ConvectionDiffusionReactionFormulation::u_dirichlet()
{
  if (_formulationChoice == ULTRAWEAK)
  {
    return u_hat();
  }
  else
  {
    return u();
  }
}

VarPtr ConvectionDiffusionReactionFormulation::u_hat()
{
  return _vf->traceVar(S_UHAT);
}

VarPtr ConvectionDiffusionReactionFormulation::tau()
{
  if (_spaceDim > 1)
    return _vf->testVar(S_TAU, HDIV);
  else
    return _vf->testVar(S_TAU, HGRAD);
}

VarPtr ConvectionDiffusionReactionFormulation::v()
{
  return _vf->testVar(S_V, HGRAD);
}

VarFactoryPtr ConvectionDiffusionReactionFormulation::vf()
{
  return _vf;
}