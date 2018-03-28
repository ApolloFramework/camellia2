//
//  CompressibleNavierStokesFormulationRefactor.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/15/18.
//

#include "CompressibleNavierStokesFormulationRefactor.hpp"

#include "BC.h"
#include "BF.h"
#include "CompressibleNavierStokesProblem.hpp"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace std;

const string CompressibleNavierStokesFormulationRefactor::S_rho = "rho";
const string CompressibleNavierStokesFormulationRefactor::S_u1  = "u1";
const string CompressibleNavierStokesFormulationRefactor::S_u2  = "u2";
const string CompressibleNavierStokesFormulationRefactor::S_u3  = "u3";
const string CompressibleNavierStokesFormulationRefactor::S_T   = "T";
const string CompressibleNavierStokesFormulationRefactor::S_D11 = "D11";
const string CompressibleNavierStokesFormulationRefactor::S_D12 = "D12";
const string CompressibleNavierStokesFormulationRefactor::S_D13 = "D13";
const string CompressibleNavierStokesFormulationRefactor::S_D21 = "D21";
const string CompressibleNavierStokesFormulationRefactor::S_D22 = "D22";
const string CompressibleNavierStokesFormulationRefactor::S_D23 = "D23";
const string CompressibleNavierStokesFormulationRefactor::S_D31 = "D31";
const string CompressibleNavierStokesFormulationRefactor::S_D32 = "D32";
const string CompressibleNavierStokesFormulationRefactor::S_D33 = "D33";
const string CompressibleNavierStokesFormulationRefactor::S_q1 = "q1";
const string CompressibleNavierStokesFormulationRefactor::S_q2 = "q2";
const string CompressibleNavierStokesFormulationRefactor::S_q3 = "q3";

const string CompressibleNavierStokesFormulationRefactor::S_tc = "tc";
const string CompressibleNavierStokesFormulationRefactor::S_tm1 = "tm1";
const string CompressibleNavierStokesFormulationRefactor::S_tm2 = "tm2";
const string CompressibleNavierStokesFormulationRefactor::S_tm3 = "tm3";
const string CompressibleNavierStokesFormulationRefactor::S_te = "te";
const string CompressibleNavierStokesFormulationRefactor::S_u1_hat = "u1_hat";
const string CompressibleNavierStokesFormulationRefactor::S_u2_hat = "u2_hat";
const string CompressibleNavierStokesFormulationRefactor::S_u3_hat = "u3_hat";
const string CompressibleNavierStokesFormulationRefactor::S_T_hat = "T_hat";

const string CompressibleNavierStokesFormulationRefactor::S_vc = "vc";
const string CompressibleNavierStokesFormulationRefactor::S_vm1  = "vm1";
const string CompressibleNavierStokesFormulationRefactor::S_vm2  = "vm2";
const string CompressibleNavierStokesFormulationRefactor::S_vm3  = "vm3";
const string CompressibleNavierStokesFormulationRefactor::S_ve   = "ve";
const string CompressibleNavierStokesFormulationRefactor::S_S1 = "S1";
const string CompressibleNavierStokesFormulationRefactor::S_S2 = "S2";
const string CompressibleNavierStokesFormulationRefactor::S_S3 = "S3";
const string CompressibleNavierStokesFormulationRefactor::S_tau = "tau";


const string CompressibleNavierStokesFormulationRefactor::S_u[3]    = {S_u1, S_u2, S_u3};
const string CompressibleNavierStokesFormulationRefactor::S_q[3]    = {S_q1, S_q2, S_q3};
const string CompressibleNavierStokesFormulationRefactor::S_D[3][3] = {{S_D11, S_D12, S_D13},
                                                                       {S_D21, S_D22, S_D23},
                                                                       {S_D31, S_D32, S_D33}};
const string CompressibleNavierStokesFormulationRefactor::S_S[3]    = {S_S1, S_S2, S_S3};
const string CompressibleNavierStokesFormulationRefactor::S_tm[3]   = {S_tm1, S_tm2, S_tm3};
const string CompressibleNavierStokesFormulationRefactor::S_u_hat[3]= {S_u1_hat, S_u2_hat, S_u3_hat};
const string CompressibleNavierStokesFormulationRefactor::S_vm[3]   = {S_vm1, S_vm2, S_vm3};


void CompressibleNavierStokesFormulationRefactor::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}


Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> CompressibleNavierStokesFormulationRefactor::steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                                           MeshTopologyPtr meshTopo, int polyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("spatialPolyOrder", polyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesFormulationRefactor(meshTopo, parameters));
}

Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                                                 MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesFormulationRefactor(meshTopo, parameters));
}

CompressibleNavierStokesFormulationRefactor::CompressibleNavierStokesFormulationRefactor(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _spaceDim = spaceDim;
  _fc = ParameterFunction::parameterFunction(0.0);
  _fe = ParameterFunction::parameterFunction(0.0);
  _fm = vector<Teuchos::RCP<ParameterFunction> >(_spaceDim, ParameterFunction::parameterFunction(0.0));
  _mu = parameters.get<double>("mu",1.0);
  _gamma = parameters.get<double>("gamma",1.4);
  _Pr = parameters.get<double>("Pr",0.713);
  _Cv = parameters.get<double>("Cv",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string normName = parameters.get<string>("norm", "Graph");
  
  // nonlinear parameters
  bool neglectFluxesOnRHS = true; // if ever we want to support a false value here, we will need to add terms corresponding to traces/fluxes to the RHS.
  
  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double dt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  TimeStepType timeStepType = parameters.get<TimeStepType>("timeStepType", BACKWARD_EULER); // Backward Euler is immune to oscillations (which Crank-Nicolson can/does exhibit)
  
  double rhoInit = parameters.get<double>("rhoInit", 1.);
  double uInit[3];
  uInit[0] = parameters.get<double>("u1Init", 0.);
  uInit[1] = parameters.get<double>("u2Init", 0.);
  uInit[2] = parameters.get<double>("u3Init", 0.);
  double TInit = parameters.get<double>("TInit", 1.);
  
  string problemName = parameters.get<string>("problemName", "");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");
  
  _useConformingTraces = useConformingTraces;
  _spatialPolyOrder = spatialPolyOrder;
  _temporalPolyOrder =temporalPolyOrder;
  _dt = ParameterFunction::parameterFunction(dt);
  _t = ParameterFunction::parameterFunction(0);
  _t0 = parameters.get<double>("t0",0);
  _neglectFluxesOnRHS = neglectFluxesOnRHS;
  _delta_k = delta_k;
  
  _muParamFunc = ParameterFunction::parameterFunction(_mu);
  _muSqrtParamFunc = ParameterFunction::parameterFunction(sqrt(_mu));
  _muFunc = _muParamFunc;
  _muSqrtFunc = _muSqrtParamFunc;
  
  double thetaValue;
  switch (timeStepType) {
    case FORWARD_EULER:
      thetaValue = 0.0;
      break;
    case CRANK_NICOLSON:
      thetaValue = 0.5;
      break;
    case BACKWARD_EULER:
      thetaValue = 1.0;
      break;
  }
  
  _theta = ParameterFunction::parameterFunction(thetaValue);
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;
  
  // TEUCHOS_TEST_FOR_EXCEPTION(_timeStepping, std::invalid_argument, "Time stepping not supported");
  
  // field variables
  VarPtr rho;
  vector<VarPtr> u(_spaceDim);
  VarPtr T;
  vector<vector<VarPtr>> D(_spaceDim,vector<VarPtr>(_spaceDim));
  vector<VarPtr> q(_spaceDim);
  
  // trace variables
  VarPtr tc;
  vector<VarPtr> tm(_spaceDim);
  VarPtr te;
  vector<VarPtr> u_hat(_spaceDim);
  VarPtr T_hat;
  
  // test variables
  VarPtr vc;
  vector<VarPtr> vm(_spaceDim);
  VarPtr ve;
  vector<VarPtr> S(_spaceDim);
  VarPtr tau;
  
  _vf = VarFactory::varFactory();
  
  rho = _vf->fieldVar(S_rho);
  
  for (int d=0; d<spaceDim; d++)
  {
    u[d] = _vf->fieldVar(S_u[d]);
    q[d] = _vf->fieldVar(S_q[d]);
  }
  for (int d1=0; d1<spaceDim; d1++)
  {
    for (int d2=0; d2<spaceDim; d2++)
    {
      D[d1][d2] = _vf->fieldVar(S_D[d1][d2]);
    }
  }
  
  T = _vf->fieldVar(S_T);
  
  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  if (! _spaceTime)
  {
    Space uHatSpace = useConformingTraces ? HGRAD : L2;
    for (int d=0; d<spaceDim; d++)
    {
      u_hat[d] = _vf->traceVar(S_u_hat[d], one * u[d], uHatSpace);
    }
  }
  else
  {
    Space uHatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
    for (int d=0; d<spaceDim; d++)
    {
      u_hat[d] = _vf->traceVarSpaceOnly(S_u_hat[d], one * u[d], uHatSpace);
    }
  }
  
  Space THatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
  T_hat = _vf->traceVarSpaceOnly(S_T_hat, one * T, THatSpace);
  
  // FunctionPtr n = Function::normal();
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  // TODO: add in here the definitions of the LinearTerms that the fluxes below trace.
  //       (See the exactSolution_tc(), etc. methods for an indication of the right thing here.)
  tc = _vf->fluxVar(S_tc);
  for (int d=0; d<spaceDim; d++)
  {
    tm[d] = _vf->fluxVar(S_tm[d]);
  }
  te = _vf->fluxVar(S_te);
  
  vc = _vf->testVar(S_vc, HGRAD);
  for (int d=0; d<spaceDim; d++)
  {
    vm[d] = _vf->testVar(S_vm[d], HGRAD);
  }
  ve = _vf->testVar(S_ve, HGRAD);
  
  for (int d=0; d<spaceDim; d++)
  {
    Space S_space = (spaceDim == 1) ? HGRAD : HDIV;
    S[d] = _vf->testVar(S_S[d], S_space);
  }
  
  if (spaceDim == 1)  tau = _vf->testVar(S_tau, HGRAD);
  else                tau = _vf->testVar(S_tau, HDIV);
  
  // now that we have all our variables defined, process any adjustments
  map<int,VarPtr> trialVars = _vf->trialVars();
  for (auto entry : trialVars)
  {
    VarPtr var = entry.second;
    string lookupString = var->name() + "-polyOrderAdjustment";
    int adjustment = parameters.get<int>(lookupString,0);
    if (adjustment != 0)
    {
      _trialVariablePolyOrderAdjustments[var->ID()] = adjustment;
    }
  }
  
  _bf = Teuchos::rcp( new BF(_vf) );
  _rhs = RHS::rhs();
  
  vector<int> H1Order;
  if (_spaceTime)
  {
    H1Order = {spatialPolyOrder+1,temporalPolyOrder+1}; // not dead certain that temporalPolyOrder+1 is the best choice; it depends on whether the indicated poly order means L^2 as it does in space, or whether it means H^1...
  }
  else
  {
    H1Order = {spatialPolyOrder+1};
  }
  
  BCPtr bc = BC::bc();
  
  MeshPtr mesh;
  Teuchos::RCP<CompressibleNavierStokesProblem> problem;
  if (problemName != "")
  {
    problem = CompressibleNavierStokesProblem::namedProblem(problemName);
  }
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
    
    // Project ones as initial guess
    FunctionPtr rho_init, T_init;
    vector<FunctionPtr> u_init(3);
    rho_init = Function::constant(rhoInit);
    T_init = Function::constant(TInit);
    
    for (int d=0; d<spaceDim; d++)
    {
      u_init[d] = Function::constant(uInit[d]);
    }
    
    if (problem != Teuchos::null)
    {
      u_init   = problem->uInitial();
      rho_init = problem->rhoInitial();
      T_init   = problem->TInitial();
    }
    
    FunctionPtr spatialNormalTimesParity = n_x * Function::sideParity();
    
    FunctionPtr zero = Function::zero();
    FunctionPtr tc_init = zero;
    FunctionPtr te_init = zero;
    vector<FunctionPtr> tm_init(spaceDim,zero);
    
    double R = this->R();
    double Cv = this->Cv();
    
    FunctionPtr u_dot_u = zero;
    for (int d=0; d<spaceDim; d++)
    {
      u_dot_u = u_dot_u + u_init[d] * u_init[d];
    }
    for (int d=0; d<spaceDim; d++)
    {
      auto n_comp = spatialNormalTimesParity->spatialComponent(d+1);
      tc_init    = tc_init + rho_init * u_init[d] * n_comp;
      te_init = te_init + ((Cv + R) * T_init + 0.5 * u_dot_u) * rho_init * u_init[d] * n_comp;
    }
    for (int d1=0; d1<spaceDim; d1++)
    {
      tm_init[d1] = (R * rho_init * T_init) * spatialNormalTimesParity->spatialComponent(d1+1);
      for (int d2=0; d2<spaceDim; d2++)
      {
        tm_init[d1] = tm_init[d1] + rho_init * u_init[d1] * u_init[d2] * spatialNormalTimesParity->spatialComponent(d2+1);
      }
    }
    if (_spaceTime)
    {
      FunctionPtr n_t = n_xt->t() * Function::sideParity();
      tc_init = tc_init + rho_init * n_t;
      for (int d=0; d<spaceDim; d++)
      {
        tm_init[d] = tm_init[d] + rho_init * u_init[d] * n_t;
      }
      te_init = te_init + (Cv * rho_init * T_init + 0.5 * u_dot_u)*n_t;
    }
    
    map<int, FunctionPtr> initialGuess;
    initialGuess[rho->ID()]   = rho_init;
    initialGuess[T->ID()]     = T_init;
    initialGuess[T_hat->ID()] = T_init;
    initialGuess[tc->ID()]    = tc_init;
    initialGuess[te->ID()]    = te_init;
    for (int d=0; d<spaceDim; d++)
    {
      initialGuess[u[d]->ID()]     = u_init[d];
      initialGuess[u_hat[d]->ID()] = u_init[d];
      initialGuess[tm[d]->ID()]    = tm_init[d];
    }
    
    TEUCHOS_ASSERT(_backgroundFlow->numSolutions() == 1);
    TEUCHOS_ASSERT(_solnPrevTime->numSolutions() == 1);
    const int solutionOrdinal = 0;
    
    _backgroundFlow->projectOntoMesh(initialGuess, solutionOrdinal);
    _solnPrevTime->projectOntoMesh(initialGuess, solutionOrdinal);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
    _backgroundFlow->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    _solnIncrement->loadFromHDF5(savedSolutionAndMeshPrefix+"_increment.soln");
    _solnPrevTime->loadFromHDF5(savedSolutionAndMeshPrefix+"_prevtime.soln");
  }
  
  // Previous solution values
  FunctionPtr rho_prev = Function::solution(rho, _backgroundFlow);
  FunctionPtr rho_prev_time = Function::solution(rho, _solnPrevTime);
  FunctionPtr T_prev       = Function::solution(T, _backgroundFlow);
  FunctionPtr T_prev_time  = Function::solution(T, _solnPrevTime);
  
  vector<FunctionPtr> q_prev(spaceDim);
  vector<FunctionPtr> u_prev(spaceDim);
  vector<FunctionPtr> u_prev_time(spaceDim);
  vector<vector<FunctionPtr>> D_prev(spaceDim,vector<FunctionPtr>(spaceDim));
  
  for (int d1=0; d1<spaceDim; d1++)
  {
    u_prev[d1]      = Function::solution(u[d1], _backgroundFlow);
    u_prev_time[d1] = Function::solution(u[d1], _solnPrevTime);
    q_prev[d1]      = Function::solution(q[d1], _backgroundFlow);
    for (int d2=0; d2<spaceDim; d2++)
    {
      D_prev[d1][d2] = Function::solution(D[d1][d2], _backgroundFlow);
    }
  }
  
  // S terms:
  Camellia::EOperator S_divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  auto n_S = (_spaceDim > 1) ? n_x : n_x->x();
  for (int d=0; d<spaceDim; d++)
  {
    _bf->addTerm(u[d], S[d]->applyOp(S_divOp));         // D_i = mu() * grad u_i
    _rhs->addTerm(-u_prev[d] * S[d]->applyOp(S_divOp)); // D_i = mu() * grad u_i
    for (int d2=0; d2<spaceDim; d2++)
    {
      VarPtr S_d2 = (_spaceDim > 1) ? S[d]->spatialComponent(d2+1) : S[d];
      _bf->addTerm (1./_muFunc * D[d][d2],        S_d2);
      _rhs->addTerm(-1./_muFunc * D_prev[d][d2] * S_d2);
    }
    _bf->addTerm(-u_hat[d], S[d] * n_S);
  }
  
  // tau terms:
  double Cp = this->Cp();
  double Pr = this->Pr();
  double R  = this->R();
  auto n_tau = n_S;
  Camellia::EOperator tauDivOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  
  _bf->addTerm (-T,       tau->applyOp(tauDivOp)); // tau = Cp*mu/Pr * grad T
  _rhs->addTerm( T_prev * tau->applyOp(tauDivOp)); // tau = Cp*_mu/Pr * grad T
  
  _bf->addTerm(T_hat,     tau * n_tau);
  for (int d=0; d<spaceDim; d++)
  {
    VarPtr tau_d = (_spaceDim > 1) ? tau->spatialComponent(d+1) : tau;
    _bf->addTerm ( Pr/(Cp*_muFunc) * q[d],       tau_d);
    _rhs->addTerm(-Pr/(Cp*_muFunc) * q_prev[d] * tau_d);
  }

  // vc terms:
  if (_spaceTime)
  {
    _bf->addTerm (-rho,       vc->dt());
    _rhs->addTerm( rho_prev * vc->dt());
  }
  else if (_timeStepping)
  {
    cout << "timestepping" << endl;
    _bf->addTerm (rho/_dt,            vc); // TODO: add theta weight here and anywhere else it should go...
//    FunctionPtr rho_prev_time_dt = rho_prev_time/_dt;
    _rhs->addTerm(rho_prev_time/(FunctionPtr)_dt  * vc); // cast _dt to FunctionPtr because otherwise compiler complains of operator overloading ambiguity
    _rhs->addTerm(-rho_prev/(FunctionPtr)_dt * vc); // ditto here
  }
  for (int d=0; d<spaceDim; d++)
  {
    _bf->addTerm(-(u_prev[d] * rho + rho_prev * u[d]), vc->di(d+1));
    _rhs->addTerm( rho_prev * u_prev[d] * vc->di(d+1));
  }
  _bf->addTerm(tc, vc);
  _rhs->addTerm(FunctionPtr(_fc) * vc);
  
  // D is the
  double D_traceWeight = -2./3.; // In Truman's code, this is hard-coded to -2/3 for 1D, 3D, and -2/2 for 2D.  This value arises from Stokes' hypothesis, and I think he probably was implementing a variant of this for 2D.  I'm going with what I think is the more standard choice of using the same value regardless of spatial dimension.
  // vm
  for (int d1=0; d1<spaceDim; d1++)
  {
    if (_spaceTime)
    {
      _bf->addTerm(-(u_prev[d1]*rho+rho_prev*u[d1]), vm[d1]->dt());
      _rhs->addTerm( rho_prev*u_prev[d1] * vm[d1]->dt() );
    }
    if (_timeStepping)
    {
      _bf->addTerm( (u_prev[d1]*rho+rho_prev*u[d1]),   vm[d1]); // TODO: add theta weight here and anywhere else it should go...
      
      _rhs->addTerm( rho_prev_time * u_prev_time[d1] * vm[d1] );
      _rhs->addTerm(-rho_prev      * u_prev[d1]      * vm[d1] );
    }
    _bf->addTerm(-R * T_prev * rho, vm[d1]->di(d1+1));
    _bf->addTerm(-R * rho_prev * T, vm[d1]->di(d1+1));
    _rhs->addTerm(R * rho_prev * T_prev * vm[d1]->di(d1+1));
    for (int d2=0; d2<spaceDim; d2++)
    {
      _bf->addTerm(-u_prev[d1]*u_prev[d2]*rho, vm[d1]->di(d2+1));
      _bf->addTerm(-rho_prev*u_prev[d1]*u[d2], vm[d1]->di(d2+1));
      _bf->addTerm(-rho_prev*u_prev[d2]*u[d1], vm[d1]->di(d2+1));
      _rhs->addTerm( rho_prev*u_prev[d1]*u_prev[d2] * vm[d1]->di(d2+1));

      _bf->addTerm(D[d1][d2] + D[d2][d1], vm[d1]->di(d2+1));
      _bf->addTerm(D_traceWeight * D[d2][d2], vm[d1]->di(d1+1));
      
      _rhs->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1]) * vm[d1]->di(d2+1));
      _rhs->addTerm(-D_traceWeight * D_prev[d2][d2] * vm[d1]->di(d1+1));
    }
    _bf->addTerm(tm[d1], vm[d1]);
    _rhs->addTerm(FunctionPtr(_fm[d1]) * vm[d1]);
  }
  
  // ve:
  double Cv = this->Cv();
  if (_spaceTime)
  {
    _bf->addTerm(-(Cv*T_prev*rho+Cv*rho_prev*T), ve->dt());
    _rhs->addTerm(Cv*rho_prev*T_prev * ve->dt());
  }
  if (_timeStepping)
  {
    _rhs->addTerm(Cv*rho_prev_time*T_prev_time * ve);
    _rhs->addTerm(-Cv*rho_prev*T_prev * ve);
  }
  for (int d1=0; d1<spaceDim; d1++)
  {
    if (_spaceTime)
    {
      _bf->addTerm(-0.5*(u_prev[d1]*u_prev[d1])*rho, ve->dt());
      _bf->addTerm((-rho_prev*u_prev[d1])*u[d1],     ve->dt());
      
      _rhs->addTerm(0.5*rho_prev*(u_prev[d1]*u_prev[d1]) * ve->dt());
    }
    if (_timeStepping)
    {
      _bf->addTerm( 0.5*(u_prev[d1]*u_prev[d1])*rho, ve);
      _bf->addTerm( (rho_prev*u_prev[d1])*u[d1], ve);
      
      _rhs->addTerm(0.5*rho_prev_time*(u_prev_time[d1]*u_prev_time[d1]) * ve);
      _rhs->addTerm(-0.5*rho_prev*(u_prev[d1]*u_prev[d1]) * ve);
    }
    
    _bf->addTerm(-(Cv+R) * u_prev[d1] * T_prev * rho - (Cv+R) * rho_prev * T_prev * u[d1] - (Cv+R) * rho_prev * u_prev[d1] * T, ve->di(d1+1));
    _rhs->addTerm((Cv+R) * rho_prev * u_prev[d1] * T_prev * ve->di(d1+1));
    
    for (int d2=0; d2<spaceDim; d2++)
    {
      _bf->addTerm(-(0.5*(u_prev[d2]*u_prev[d2])*u_prev[d1]*rho), ve->di(d1+1));
      _bf->addTerm(-(0.5*rho_prev*(u_prev[d2]*u_prev[d2]))*u[d1], ve->di(d1+1));
      _bf->addTerm(-(rho_prev*u_prev[d1]*(u_prev[d2]*u[d2])), ve->di(d1+1));
      
      _rhs->addTerm(0.5 * rho_prev * (u_prev[d2]*u_prev[d2]) * u_prev[d1] * ve->di(d1+1));
    }
    
    _bf->addTerm(-q[d1], ve->di(d1+1));
    _rhs->addTerm(q_prev[d1] * ve->di(d1+1));

    for (int d2=0; d2<spaceDim; d2++)
    {
      _bf->addTerm((D_prev[d1][d2] + D_prev[d2][d1])*u[d2], ve->di(d1+1));
      _bf->addTerm((D_traceWeight * D_prev[d2][d2])*u[d1], ve->di(d1+1));
      _bf->addTerm((D[d1][d2] + D[d2][d1])*u_prev[d2], ve->di(d1+1));
      _bf->addTerm((D_traceWeight * u_prev[d1]) * D[d2][d2], ve->di(d1+1));
      
      _rhs->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1]) * u_prev[d2] * ve->di(d1+1));
      _rhs->addTerm(- D_traceWeight * D_prev[d2][d2] * u_prev[d1] * ve->di(d1+1));
    }
  }
  _bf->addTerm(te, ve);
  _rhs->addTerm(FunctionPtr(_fe) * ve);
  
  vector<VarPtr> missingTestVars = _bf->missingTestVars();
  vector<VarPtr> missingTrialVars = _bf->missingTrialVars();
  for (int i=0; i < missingTestVars.size(); i++)
  {
    VarPtr var = missingTestVars[i];
    cout << var->displayString() << endl;
  }
  for (int i=0; i < missingTrialVars.size(); i++)
  {
    VarPtr var = missingTrialVars[i];
    cout << var->displayString() << endl;
  }
  
  // TODO: consider adding support for Truman's various IP definitions
  // (For now, we just support the graph norm.)
  
  TEUCHOS_TEST_FOR_EXCEPTION(normName != "Graph", std::invalid_argument, "non-graph norms not yet supported in the refactor; use legacy instead");
  _ips["Graph"] = _bf->graphNorm();
  IPPtr ip = _ips.at(normName);
  
  
  // _solnIncrement->setBC(bc);
  _solnIncrement->setRHS(_rhs);
  _solnIncrement->setIP(ip);
  // _solnIncrement->setRHS(rhs);
  
  mesh->registerSolution(_backgroundFlow);
  mesh->registerSolution(_solnIncrement);
  mesh->registerSolution(_solnPrevTime);
  
  // LinearTermPtr residual = rhs->linearTerm() - _bf->testFunctional(_solnIncrement,true); // false: don't exclude boundary terms
  // LinearTermPtr residual = _rhsForResidual->linearTerm() - _bf->testFunctional(_solnIncrement,false); // false: don't exclude boundary terms
  // LinearTermPtr residual = _rhsForSolve->linearTerm() - _bf->testFunctional(_solnIncrement,true); // false: don't exclude boundary terms
  
  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );
  // _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, _ips[normName], energyThreshold ) );
  
  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );
  
  // Set up Functions for L^2 norm computations
  
  FunctionPtr rho_incr = Function::solution(rho, _solnIncrement);
  FunctionPtr T_incr = Function::solution(T, _solnIncrement);
  
  _L2IncrementFunction = rho_incr * rho_incr + T_incr * T_incr;
  _L2SolutionFunction = rho_prev * rho_prev + T_prev * T_prev;
  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    FunctionPtr u_i_incr = Function::solution(this->u(comp_i), _solnIncrement);
    FunctionPtr u_i_prev = Function::solution(this->u(comp_i), _backgroundFlow);
    FunctionPtr q_i_incr = Function::solution(this->q(comp_i), _solnIncrement);
    FunctionPtr q_i_prev = Function::solution(this->q(comp_i), _backgroundFlow);
    
    _L2IncrementFunction = _L2IncrementFunction + u_i_incr * u_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + u_i_prev * u_i_prev;
    _L2IncrementFunction = _L2IncrementFunction + q_i_incr * q_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + q_i_prev * q_i_prev;
    
    for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
    {
      FunctionPtr D_ij_incr = Function::solution(this->D(comp_i,comp_j), _solnIncrement);
      FunctionPtr D_ij_prev = Function::solution(this->D(comp_i,comp_j), _backgroundFlow);
      _L2IncrementFunction = _L2IncrementFunction + D_ij_incr * D_ij_incr;
      _L2SolutionFunction = _L2SolutionFunction + D_ij_prev * D_ij_prev;
    }
  }
  
  _solver = Solver::getDirectSolver();
  
  _nonlinearIterationCount = 0;
}

void CompressibleNavierStokesFormulationRefactor::addVelocityTraceComponentCondition(SpatialFilterPtr region, FunctionPtr ui_exact, int i)
{
  VarPtr ui_hat = this->u_hat(i);
  _solnIncrement->bc()->addDirichlet(ui_hat, region, ui_exact);
}


void CompressibleNavierStokesFormulationRefactor::addVelocityTraceCondition(SpatialFilterPtr region, FunctionPtr u_exact)
{
  if (_spaceDim==1)
    addVelocityTraceComponentCondition(region, u_exact, 1);
  else
  {
    for (int d=0; d<_spaceDim; d++)
    {
      addVelocityTraceComponentCondition(region, u_exact->spatialComponent(d+1), d+1);
    }
  }
}

void CompressibleNavierStokesFormulationRefactor::addTemperatureTraceCondition(SpatialFilterPtr region, FunctionPtr T_exact)
{
  VarPtr T_hat = this->T_hat();
  _solnIncrement->bc()->addDirichlet(T_hat, region, T_exact);
}

void CompressibleNavierStokesFormulationRefactor::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr tc_exact)
{
  VarPtr tc = this->tc();
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
}

void CompressibleNavierStokesFormulationRefactor::addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr tm_i_exact, int i)
{
  VarPtr tm_i = this->tm(i);
  _solnIncrement->bc()->addDirichlet(tm_i, region, tm_i_exact);
}

void CompressibleNavierStokesFormulationRefactor::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr te_exact)
{
  VarPtr te = this->te();
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
}

void CompressibleNavierStokesFormulationRefactor::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr tc = this->tc();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tc_exact = this->exactSolution_tc(u_exact, rho_exact, T_exact, includeParity);
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
}

void CompressibleNavierStokesFormulationRefactor::addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact, int i)
{
  VarPtr tm_i = this->tm(i);
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  FunctionPtr tm_i_exact = exactSolution_tm(u_exact, rho_exact, T_exact, includeParity)[i-1];
  _solnIncrement->bc()->addDirichlet(tm_i, region, tm_i_exact);
}

void CompressibleNavierStokesFormulationRefactor::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tm_exact = exactSolution_tm(u_exact, rho_exact, T_exact, includeParity);
  for (int d=0; d<_spaceDim; d++)
  {
    VarPtr tm_i = this->tm(d+1);
    _solnIncrement->bc()->addDirichlet(tm_i, region, tm_exact[d]);
  }
}

void CompressibleNavierStokesFormulationRefactor::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr te = this->te();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto te_exact = exactSolution_te(u_exact, rho_exact, T_exact, includeParity);
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
}

BFPtr CompressibleNavierStokesFormulationRefactor::bf()
{
  return _bf;
}

double CompressibleNavierStokesFormulationRefactor::Cv()
{
  return _Cv;
}

double CompressibleNavierStokesFormulationRefactor::Cp()
{
  return _gamma*_Cv;
}

VarPtr CompressibleNavierStokesFormulationRefactor::D(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  
  return _vf->fieldVar(S_D[i-1][j-1]);
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the continuity equation
FunctionPtr CompressibleNavierStokesFormulationRefactor::exactSolution_fc(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  // strong form of the equation has
  // d/dt rho + div ( rho u ) = f_c
  FunctionPtr f_c = Function::zero();
  if (rho->dt() != Teuchos::null)
  {
    f_c = f_c + rho->dt();
  }
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f_c = f_c + Function::op(rho * u, divOp);
  return f_c;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
FunctionPtr CompressibleNavierStokesFormulationRefactor::exactSolution_fe(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  bool includeParity = false; // we don't use any traces or fluxes below, so this does not matter
  auto exactMap = this->exactSolutionMap(u, rho, T, includeParity);
  // strong form of the equation has
  // f_e =   d/dt ( rho * ( c_v T + 0.5 * u * u) )
  //       + div  ( rho * u * ( c_v T + 0.5 * u * u) + rho * u * R * T + q - u \dot (D + D^T - 2/3 tr(D) I) )
  // define g_e = rho * ( c_v T + 0.5 * u * u)
  FunctionPtr f_e = Function::zero();
  double c_v = this->Cv();
  double R   = this->R();
  FunctionPtr g_e = rho * ( c_v * T + 0.5 * u * u);
  if (g_e->dt() != Teuchos::null)
  {
    f_e = f_e + g_e->dt();
  }
  FunctionPtr q;
  if (_spaceDim == 1)
    q = exactMap[this->q(1)->ID()];
  else
  {
    vector<FunctionPtr> q_vector(_spaceDim);
    for (int d=0; d<_spaceDim; d++)
    {
      q_vector[d] = exactMap[this->q(d+1)->ID()];
    }
    q = (_spaceDim > 1) ? Function::vectorize(q_vector) : q_vector[0];
  }
  FunctionPtr D_trace = Function::zero();
  for (int d=0; d<_spaceDim; d++)
  {
    VarPtr D_dd = this->D(d+1,d+1);
    D_trace = D_trace + exactMap[D_dd->ID()];
  }
  FunctionPtr u_dot_sigma = Function::zero();
  for (int d1=0; d1<_spaceDim; d1++)
  {
    int i = d1+1;
    FunctionPtr u_i = exactMap[this->u(i)->ID()];
    vector<FunctionPtr> sigmaRow_vector(_spaceDim); // sigma_i
    for (int d2=0; d2<_spaceDim; d2++)
    {
      int j = d2 + 1;
      FunctionPtr D_ij = exactMap[this->D(i,j)->ID()];
      FunctionPtr D_ji = exactMap[this->D(j,i)->ID()];
      FunctionPtr D_trace_contribution = (i==j)? -2./3. * D_trace : Function::zero();
      sigmaRow_vector[d2] = D_ij + D_ji + D_trace_contribution;
    }
    FunctionPtr sigmaRow =  (_spaceDim > 1) ? Function::vectorize(sigmaRow_vector) : sigmaRow_vector[0];
    u_dot_sigma = u_dot_sigma + u_i * sigmaRow;
  }
  FunctionPtr flux_term = u * g_e + rho * R * T * u + q - u_dot_sigma;
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f_e = f_e + Function::op(flux_term, divOp);
  return f_e;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> CompressibleNavierStokesFormulationRefactor::exactSolution_fm(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  bool includeParity = false; // we don't use any traces or fluxes below, so this does not matter
  auto exactMap = this->exactSolutionMap(u, rho, T, includeParity);
  double R   = this->R();
  vector<FunctionPtr> f_m(_spaceDim, Function::zero());
  FunctionPtr rho_u = rho * u;
  if (rho_u->dt() != Teuchos::null)
  {
    if (_spaceDim == 1)
    {
      f_m[0] = f_m[0] + rho_u->dt();
    }
    else
    {
      FunctionPtr dt_part = rho_u->dt();
      for (int d=0; d<_spaceDim; d++)
      {
        f_m[d] = f_m[d] + dt_part->spatialComponent(d+1);
      }
    }
  }
  vector<FunctionPtr> u_vector(_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    u_vector[d] = exactMap[this->u(d+1)->ID()];
  }

  for (int d2=0; d2<_spaceDim; d2++)
  {
    int j = d2+1;
    vector<FunctionPtr> column_vector(_spaceDim, Function::zero());
    for (int d1=0; d1<_spaceDim; d1++)
    {
      int i = d1+1;
      column_vector[d1] = rho * u_vector[d1] * u_vector[d2];
      if (i==j)
      {
        column_vector[d1] = column_vector[d1] + R * (rho * T);
      }
    }
    FunctionPtr column = (_spaceDim > 1) ? Function::vectorize(column_vector) : column_vector[0];
    Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    f_m[d2] = f_m[d2] + Function::op(column, divOp);
  }
  return f_m;
}

FunctionPtr CompressibleNavierStokesFormulationRefactor::exactSolution_tc(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  FunctionPtr tc_exact = Function::zero();
  for (int d=0; d<_spaceDim; d++)
  {
    auto u_i = (_spaceDim > 1) ? velocity->spatialComponent(d+1) : velocity;
    auto n_i = n->spatialComponent(d+1);
    tc_exact = tc_exact + rho * u_i * n_i;
  }
  if (_spaceTime)
  {
    FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
    FunctionPtr n_t = n_xt->t();
    tc_exact = tc_exact + rho * n_t;
  }
  if (includeParity)
  {
    tc_exact = tc_exact * Function::sideParity();
  }
  return tc_exact;
}

FunctionPtr CompressibleNavierStokesFormulationRefactor::exactSolution_te(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  // t_e is the trace of:
  // ((c_v + R) T rho u + 0.5 * (u dot u) rho u + q - u dot (D + D^T - 2/3 tr(D) I)) dot n
  
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  FunctionPtr D_trace = Function::zero();
  std::vector<FunctionPtr> Di_exact(_spaceDim);
  std::vector<std::vector<FunctionPtr>> Dij_exact(_spaceDim, std::vector<FunctionPtr>(_spaceDim));
  std::vector<FunctionPtr> u_vector(_spaceDim);
  FunctionPtr q_exact;
  
  FunctionPtr qWeight = (-Cp()/Pr())*_muFunc;
  if (_spaceDim == 1)
  {
    q_exact = qWeight * T->dx();
  }
  else
  {
    q_exact = qWeight * T->grad();
  }
  
  for (int d=0; d<_spaceDim; d++)
  {
    if (_spaceDim == 1)
    {
      u_vector[d] = velocity;
      Di_exact[d] = _mu * u_vector[d]->dx();
      D_trace = Di_exact[d];
      Dij_exact[0][0] = Di_exact[0];
    }
    else
    {
      u_vector[d] = velocity->spatialComponent(d+1);
      Di_exact[d] = _mu * u_vector[d]->grad();
      D_trace = D_trace + Di_exact[d]->spatialComponent(d+1);
      for (int d2=0; d2<_spaceDim; d2++)
      {
        Dij_exact[d][d2] = Di_exact[d]->spatialComponent(d2+1);
      }
    }
  }
  
  double R = this->R();
  double Cv = this->Cv();
  double D_traceWeight = -2./3.; // Stokes' hypothesis
  
  // defer the dotting with the normal until we've accumulated the other terms (other than the D + D^T terms, which we treat separately)
  FunctionPtr te_exact = ((Cv + R) * T + (0.5 * velocity * velocity)) * rho * velocity + q_exact;
  te_exact = te_exact - D_traceWeight * D_trace * velocity; // simplification of u dot (2/3 tr(D) I)
  
  // now, dot with normal
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  if (_spaceDim == 1)
  {
    // for 1D, the product with normal should yield a scalar result
    te_exact = te_exact * n->x();
  }
  else
  {
    te_exact = te_exact * n; // dot product
  }
  // u dot (D + D^T) dot n = ((D + D^T) u) dot n
  for (int d1=0; d1<_spaceDim; d1++)
  {
    for (int d2=0; d2<_spaceDim; d2++)
    {
      te_exact = te_exact - (Dij_exact[d1][d2] + Dij_exact[d2][d1]) * u_vector[d2] * n->spatialComponent(d1+1);
    }
  }
  if (includeParity)
  {
    te_exact = te_exact * Function::sideParity();
  }
  return te_exact;
}

std::vector<FunctionPtr> CompressibleNavierStokesFormulationRefactor::exactSolution_tm(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  vector<FunctionPtr> tm_exact(_spaceDim);
  for (int i=1; i<= _spaceDim; i++)
  {
    VarPtr tm_i = this->tm(i);
    
    // tm: trace of rho (u xx u) n + rho R T I n - (D + D^T - 2./3. * tr(D)I) n
    //     (where xx is the outer product operator)
    
    FunctionPtr n = TFunction<double>::normal(); // spatial normal
    
    FunctionPtr D_trace = Function::zero();
    std::vector<FunctionPtr> Di_exact(_spaceDim);
    std::vector<std::vector<FunctionPtr>> Dij_exact(_spaceDim, std::vector<FunctionPtr>(_spaceDim));
    std::vector<FunctionPtr> u_vector(_spaceDim);
    for (int d=0; d<_spaceDim; d++)
    {
      if (_spaceDim == 1)
      {
        u_vector[d] = velocity;
        Di_exact[d] = _mu * u_vector[d]->dx();
        D_trace = Di_exact[d];
        Dij_exact[0][0] = Di_exact[0];
      }
      else
      {
        u_vector[d] = velocity->spatialComponent(d+1);
        Di_exact[d] = _mu * u_vector[d]->grad();
        D_trace = D_trace + Di_exact[d]->spatialComponent(d+1);
        for (int d2=0; d2<_spaceDim; d2++)
        {
          Dij_exact[d][d2] = Di_exact[d]->spatialComponent(d2+1);
        }
      }
    }
    
    double R = this->R();
    double D_traceWeight = -2./3.; // Stokes' hypothesis
    
    FunctionPtr tm_i_exact = Function::zero();
    for (int d=0; d<_spaceDim; d++)
    {
      // rho (u xx u) n
      tm_i_exact = tm_i_exact + rho * u_vector[d] * u_vector[i-1] * n->spatialComponent(d+1);
      // - (D + D^T) n
      tm_i_exact = tm_i_exact - (Dij_exact[d][i-1] + Dij_exact[i-1][d]) * n->spatialComponent(d+1);
    }
    // rho R T I n
    tm_i_exact = tm_i_exact + rho * R * T * n->spatialComponent(i);
    // - D_traceWeight * tr(D) I n
    tm_i_exact = tm_i_exact - D_traceWeight * D_trace * n->spatialComponent(i);
    
    if (_spaceTime)
    {
      // TODO: confirm that this is correct (I'm not really focused on the space-time case in this refactor...)
      FunctionPtr n_t = Function::normalSpaceTime()->t();
      FunctionPtr u_i = u_vector[i-1];
      tm_i_exact = tm_i_exact + rho * u_i * n_t;
    }
    
    if (includeParity)
    {
      tm_i_exact = tm_i_exact * Function::sideParity();
    }
    tm_exact[i-1] = tm_i_exact;
  }
  return tm_exact;
}

std::map<int, FunctionPtr> CompressibleNavierStokesFormulationRefactor::exactSolutionMap(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeFluxParity)
{
  using namespace std;
  vector<FunctionPtr> q(_spaceDim);
  vector<vector<FunctionPtr>> D(_spaceDim,vector<FunctionPtr>(_spaceDim));
  vector<FunctionPtr> u(_spaceDim);
  FunctionPtr qWeight = (-Cp()/Pr())*_muFunc;
  if (_spaceDim == 1)
  {
    D[0][0] = _muFunc * velocity->dx();
    q[0] = qWeight * T->dx();
    u[0] = velocity;
  }
  else
  {
    for (int d1=0; d1<_spaceDim; d1++)
    {
      q[d1] = qWeight * T->di(d1+1);
      u[d1] = velocity->spatialComponent(d1+1);
      for (int d2=0; d2<_spaceDim; d2++)
      {
        D[d1][d2] = _muFunc * u[d1]->di(d2+1);
      }
    }
  }
  vector<FunctionPtr> tm = exactSolution_tm(velocity, rho, T, includeFluxParity);
  FunctionPtr         te = exactSolution_te(velocity, rho, T, includeFluxParity);
  FunctionPtr         tc = exactSolution_tc(velocity, rho, T, includeFluxParity);
  
  map<int, FunctionPtr> solnMap;
  solnMap[this->T()->ID()]     = T;
  solnMap[this->T_hat()->ID()] = T;
  solnMap[this->rho()->ID()]   = rho;
  solnMap[this->tc()->ID()]    = tc;
  solnMap[this->te()->ID()]    = te;
  for (int d1=0; d1<_spaceDim; d1++)
  {
    solnMap[this->u(d1+1)->ID()]     = u[d1];
    solnMap[this->u_hat(d1+1)->ID()] = u[d1];
    
    solnMap[this->q(d1+1)->ID()]     = q[d1];
    
    solnMap[this->tm(d1+1)->ID()]    = tm[d1];
    
    for (int d2=0; d2<_spaceDim; d2++)
    {
      solnMap[this->D(d1+1,d2+1)->ID()] = D[d1][d2];
    }
  }
  return solnMap;
}

double CompressibleNavierStokesFormulationRefactor::gamma()
{
  return _gamma;
}

int CompressibleNavierStokesFormulationRefactor::getSolveCode()
{
  return _solveCode;
}

double CompressibleNavierStokesFormulationRefactor::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double CompressibleNavierStokesFormulationRefactor::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

double CompressibleNavierStokesFormulationRefactor::mu()
{
  return _mu;
}

double CompressibleNavierStokesFormulationRefactor::Pr()
{
  return _Pr;
}

VarPtr CompressibleNavierStokesFormulationRefactor::q(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_q[i-1]);
}

double CompressibleNavierStokesFormulationRefactor::R()
{
  return Cp()-Cv();
}

VarPtr CompressibleNavierStokesFormulationRefactor::rho()
{
  return _vf->fieldVar(S_rho);
}

RHSPtr CompressibleNavierStokesFormulationRefactor::rhs()
{
  return _rhs;
}

VarPtr CompressibleNavierStokesFormulationRefactor::S(int i)
{
  CHECK_VALID_COMPONENT(i);
  Space SSpace = (_spaceDim == 1) ? HGRAD : HDIV;
  return _vf->testVar(S_S[i-1], SSpace);
}

void CompressibleNavierStokesFormulationRefactor::setForcing(FunctionPtr f_continuity, vector<FunctionPtr> f_momentum, FunctionPtr f_energy)
{
  TEUCHOS_TEST_FOR_EXCEPTION(f_momentum.size() != _spaceDim, std::invalid_argument, "f_momentum should have size equal to the spatial dimension");
  _fc->setValue(f_continuity);
  _fe->setValue(f_energy);
  for (int d=0; d<_spaceDim; d++)
  {
    _fm[d]->setValue(f_momentum[d]);
  }
}

// ! set current time step used for transient solve
void CompressibleNavierStokesFormulationRefactor::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double CompressibleNavierStokesFormulationRefactor::solveAndAccumulate()
{
  _solveCode = _solnIncrement->solve(_solver);
  
  set<int> nonlinearVariables = {{rho()->ID(), T()->ID()}};
  set<int> linearVariables = {{tc()->ID(), te()->ID(), T_hat()->ID()}};
  for (int d1=0; d1<_spaceDim; d1++)
  {
    nonlinearVariables.insert(u(d1+1)->ID());
    nonlinearVariables.insert(q(d1+1)->ID());
    linearVariables.insert(tm(d1+1)->ID());
    linearVariables.insert(u_hat(d1+1)->ID());
    for (int d2=0; d2<_spaceDim; d2++)
    {
      nonlinearVariables.insert(D(d1+1,d2+1)->ID());
    }
  }
  
  FunctionPtr rhoPrevious  = Function::solution(rho(),_backgroundFlow);
  FunctionPtr rhoIncrement = Function::solution(rho(),_solnIncrement);
  FunctionPtr TPrevious    = Function::solution(T(),  _backgroundFlow);
  FunctionPtr TIncrement   = Function::solution(T(),  _solnIncrement);
  
  vector<FunctionPtr> positiveFunctions = {rhoPrevious,  TPrevious};
  vector<FunctionPtr> positiveUpdates   = {rhoIncrement, TIncrement};
  
  double alpha = 1;
  bool useLineSearch = true;
  int posEnrich = 5;
  if (useLineSearch)
  {
    double lineSearchFactor = .5;
    double eps = .001;
    bool isPositive=true;
    for (int i=0; i < positiveFunctions.size(); i++)
    {
      FunctionPtr temp = positiveFunctions[i] + alpha*positiveUpdates[i] - Function::constant(eps);
      isPositive = isPositive and temp->isPositive(_solnIncrement->mesh(),posEnrich);
    }
    int iter = 0; int maxIter = 20;
    while (!isPositive && iter < maxIter)
    {
      alpha = alpha*lineSearchFactor;
      isPositive = true;
      for (int i=0; i < positiveFunctions.size(); i++)
      {
        FunctionPtr temp = positiveFunctions[i] + alpha*positiveUpdates[i] - Function::constant(eps);
        isPositive = isPositive and temp->isPositive(_solnIncrement->mesh(),posEnrich);
      }
      iter++;
    }
    int commRank = Teuchos::GlobalMPISession::getRank();
    // if (commRank==0 && alpha < 1.0){
    //   cout << "Line search factor alpha = " << alpha << endl;
    // }
  }
  
  _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, nonlinearVariables, linearVariables);
  _nonlinearIterationCount++;
  
  return alpha;
}

// ! Returns the solution (at current time)
SolutionPtr CompressibleNavierStokesFormulationRefactor::solution()
{
  return _backgroundFlow;
}

SolutionPtr CompressibleNavierStokesFormulationRefactor::solutionIncrement()
{
  return _solnIncrement;
}

VarPtr CompressibleNavierStokesFormulationRefactor::T()
{
  return _vf->fieldVar(S_T);
}

VarPtr CompressibleNavierStokesFormulationRefactor::T_hat()
{
  if (! _spaceTime)
    return _vf->traceVar(S_T_hat);
  else
    return _vf->traceVarSpaceOnly(S_T_hat);
}

VarPtr CompressibleNavierStokesFormulationRefactor::tau()
{
  Space tauSpace = (_spaceDim == 1) ? HGRAD : HDIV;
  return _vf->testVar(S_tau, tauSpace);
}

VarPtr CompressibleNavierStokesFormulationRefactor::tc()
{
  return _vf->fluxVar(S_tc);
}

VarPtr CompressibleNavierStokesFormulationRefactor::te()
{
  return _vf->fluxVar(S_te);
}

VarPtr CompressibleNavierStokesFormulationRefactor::tm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->fluxVar(S_tm[i-1]);
}

VarPtr CompressibleNavierStokesFormulationRefactor::u(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_u[i-1]);
}

// traces:
VarPtr CompressibleNavierStokesFormulationRefactor::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  if (! _spaceTime)
    return _vf->traceVar(S_u_hat[i-1]);
  else
    return _vf->traceVarSpaceOnly(S_u_hat[i-1]);
}

VarPtr CompressibleNavierStokesFormulationRefactor::vc()
{
  return _vf->testVar(S_vc, HGRAD);
}

VarPtr CompressibleNavierStokesFormulationRefactor::vm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vm[i-1], HGRAD);
}

VarPtr CompressibleNavierStokesFormulationRefactor::ve()
{
  return _vf->testVar(S_ve, HGRAD);
}
