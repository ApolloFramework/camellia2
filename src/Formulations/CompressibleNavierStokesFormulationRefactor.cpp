//
//  CompressibleNavierStokesFormulationRefactorRefactor.cpp
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
const string CompressibleNavierStokesFormulationRefactor::S_u[3] = {S_u1, S_u2, S_u3};
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

const string CompressibleNavierStokesFormulationRefactor::S_q[3]    = {S_q1, S_q2, S_q3};
const string CompressibleNavierStokesFormulationRefactor::S_D[3][3] = {{S_D11, S_D12, S_D13},
                                                                       {S_D21, S_D22, S_D23},
                                                                       {S_D31, S_D32, S_D33}};
const string CompressibleNavierStokesFormulationRefactor::S_tm[3]   = {S_tm1, S_tm2, S_tm3};
const string CompressibleNavierStokesFormulationRefactor::S_u_hat[3]= {S_u1_hat, S_u2_hat, S_u3_hat};
const string CompressibleNavierStokesFormulationRefactor::S_vm[3]   = {S_vm1, S_vm2, S_vm3};

CompressibleNavierStokesFormulationRefactor CompressibleNavierStokesFormulationRefactor::steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
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
  
  return CompressibleNavierStokesFormulationRefactor(meshTopo, parameters);
}

CompressibleNavierStokesFormulationRefactor CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
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
  
  return CompressibleNavierStokesFormulationRefactor(meshTopo, parameters);
}

CompressibleNavierStokesFormulationRefactor::CompressibleNavierStokesFormulationRefactor(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
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
  bool neglectFluxesOnRHS = true;
  
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
  
  string problemName = parameters.get<string>("problemName", "Trivial");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");
  
  _spaceDim = spaceDim;
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
  
  // Too complicated at the moment to define where these other trace variables come from
  tc = _vf->fluxVar(S_tc);
  for (int d=0; d<spaceDim; d++)
  {
    tm[d] = _vf->fluxVar(S_tm[d]);
  }
  te = _vf->fluxVar(S_te);
  
  vc = _vf->testVar(S_vc, HGRAD);
  for (int d=0; d<spaceDim; d++)
  {
    vm[d] = _vf->fluxVar(S_vm[d]);
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
      auto n_comp = spatialNormalTimesParity->spatialComponent(d);
      tc_init    = tc_init + rho_init * u_init[d] * n_comp;
      te_init = te_init + ((Cv + R) * T_init + 0.5 * u_dot_u) * rho_init * u_init[d] * n_comp;
    }
    for (int d1=0; d1<spaceDim; d1++)
    {
      tm_init[d1] = (R * rho_init * T_init) * spatialNormalTimesParity->spatialComponent(d1);
      for (int d2=0; d2<spaceDim; d2++)
      {
        tm_init[d1] = tm_init[d1] + rho_init * u_init[d1] * u_init[d2] * spatialNormalTimesParity->spatialComponent(d2);
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
  
  vector<FunctionPtr> q_prev;
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
  for (int d=0; d<spaceDim; d++)
  {
    _bf->addTerm(u[d], S[d]->div());         // D_i = mu() * grad u_i
    _rhs->addTerm(-u_prev[d] * S[d]->div()); // D_i = mu() * grad u_i
    for (int d2=0; d2<spaceDim; d2++)
    {
      _bf->addTerm (1./_muFunc * D[d][d2],        S[d]->spatialComponent(d2));
      _rhs->addTerm(-1./_muFunc * D_prev[d][d2] * S[d]->spatialComponent(d2));
    }
    _bf->addTerm(-u_hat[d], S[d] * n_x);
  }
  
  // tau terms:
  double Cp = this->Cp();
  double Pr = this->Pr();
  double R  = this->R();
  if (spaceDim == 1)
  {
    _bf->addTerm (-T,      tau->dx()); // tau = Cp*mu/Pr * grad T
    _rhs->addTerm(T_prev * tau->dx()); // tau = Cp*_mu/Pr * grad T
  }
  else
  {
    _bf->addTerm (-T,       tau->div()); // tau = Cp*mu/Pr * grad T
    _rhs->addTerm( T_prev * tau->div()); // tau = Cp*_mu/Pr * grad T
  }
  _bf->addTerm(T_hat,     tau*n_x);
  for (int d=0; d<spaceDim; d++)
  {
    _bf->addTerm ( Pr/(Cp*_muFunc) * q[d],       tau->spatialComponent(d));
    _rhs->addTerm(-Pr/(Cp*_muFunc) * q_prev[d] * tau->spatialComponent(d));
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
    FunctionPtr rho_prev_time_dt = rho_prev_time/_dt; // separated onto its own line because otherwise compiler complains of operator overloading ambiguity
    _rhs->addTerm(rho_prev_time_dt  * vc);
    FunctionPtr minus_rho_prev_dt = -rho_prev/_dt; // ditto here (ambiguity in operator*)
    _rhs->addTerm(minus_rho_prev_dt * vc);
  }
  for (int d=0; d<spaceDim; d++)
  {
    _bf->addTerm(-(u_prev[d] * rho + rho_prev * u[d]), vc->di(d));
    _rhs->addTerm( rho_prev * u_prev[d] * vc->di(d));
  }
  _bf->addTerm(tc, vc);
  
  // I believe D is the stress tensor, sigma.  I'd like to rename it once I confirm this.
  double D_traceWeight; // In Truman's code, this is hard-coded to -2/3 for 1D, 3D, and -2/2 for 2D.  For now, I accept these values, but I'm suspicious.
  if (spaceDim == 2) D_traceWeight = -2./2.;
  else               D_traceWeight = -2./3.;
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
    _bf->addTerm(-R * T_prev * rho, vm[d1]->di(d1));
    _bf->addTerm(-R * rho_prev * T, vm[d1]->di(d1));
    _rhs->addTerm(R * rho_prev * T_prev * vm[d1]->di(d1));
    for (int d2=0; d2<spaceDim; d2++)
    {
      _bf->addTerm(-u_prev[d1]*u_prev[d2]*rho, vm[d1]->di(d2));
      _bf->addTerm(-rho_prev*u_prev[d1]*u[d2], vm[d1]->di(d2));
      _bf->addTerm(-rho_prev*u_prev[d2]*u[d1], vm[d1]->di(d2));
      _rhs->addTerm( rho_prev*u_prev[d1]*u_prev[d2] * vm[d1]->di(d2));

      _bf->addTerm(D[d1][d2] + D[d2][d1], vm[d1]->di(d2));
      _bf->addTerm(D_traceWeight * D[d2][d2], vm[d1]->di(d1));
      
      _rhs->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1]) * vm[d1]->di(d2));
      _rhs->addTerm(D_traceWeight * D_prev[d2][d2] * vm[d1]->di(d1));
    }
    _bf->addTerm(tm[d1], vm[d1]);
  }
  
  // ve:
  // if (_spaceTime)
  //   _bf->addTerm(-T, ve->dt());
  // _bf->addTerm(-beta_x*T + q1, ve->dx());
  // if (_spaceDim >= 2) _bf->addTerm(-beta_y*T + q2, ve->dy());
  // if (_spaceDim == 3) _bf->addTerm(-beta_z*T + q3, ve->dz());
  // _bf->addTerm(te, ve);
  switch (_spaceDim)
  {
    case 1:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*u1_prev*u1_prev*rho, ve->dt());
        _bf->addTerm(-rho_prev*u1_prev*u1, ve->dt());
      }
      if (_timeStepping)
      {
        _bf->addTerm( (Cv()*T_prev*rho+Cv()*rho_prev*T), ve);
        _bf->addTerm( 0.5*u1_prev*u1_prev*rho, ve);
        _bf->addTerm( rho_prev*u1_prev*u1, ve);
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(0.5*u1_prev*u1_prev*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*u1_prev*u1_prev*u1), ve->dx());
      _bf->addTerm(-(rho_prev*u1_prev*u1_prev*u1), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm((D11_prev+D11_prev-2./3*D11_prev)*u1, ve->dx());
      _bf->addTerm(u1_prev*(D11+D11-2./3*D11), ve->dx());
      _bf->addTerm(te, ve);
      
      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev * ve->dt());
      }
      if (_timeStepping)
      {
        _rhs->addTerm(Cv()*rho_prev_time*T_prev_time * ve);
        _rhs->addTerm(0.5*rho_prev_time*u1_prev_time*u1_prev_time * ve);
        _rhs->addTerm(-Cv()*rho_prev*T_prev * ve);
        _rhs->addTerm(-0.5*rho_prev*u1_prev*u1_prev * ve);
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev*u1_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev)*u1_prev * ve->dx());
      _rhs->addTerm(-u1_prev*(D11_prev+D11_prev-2./3*D11_prev) * ve->dx());
      break;
    case 2:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*rho, ve->dt());
        _bf->addTerm(-rho_prev*(u1_prev*u1+u2_prev*u2), ve->dt());
      }
      if (_timeStepping)
      {
        _bf->addTerm( (Cv()*T_prev*rho+Cv()*rho_prev*T), ve);
        _bf->addTerm( 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*rho, ve);
        _bf->addTerm( rho_prev*(u1_prev*u1+u2_prev*u2), ve);
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1), ve->dx());
      _bf->addTerm(-(rho_prev*u1_prev*(u1_prev*u1+u2_prev*u2)), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(Cv()*u2_prev*T_prev*rho+Cv()*rho_prev*T_prev*u2+Cv()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev*rho), ve->dy());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2), ve->dy());
      _bf->addTerm(-(rho_prev*u2_prev*(u1_prev*u1+u2_prev*u2)), ve->dy());
      _bf->addTerm(-(R()*rho_prev*T_prev*u2), ve->dy());
      _bf->addTerm(-(R()*u2_prev*T_prev*rho), ve->dy());
      _bf->addTerm(-(R()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm(-q2, ve->dy());
      _bf->addTerm((D11_prev+D11_prev-2./3*(D11_prev+D22_prev))*u1, ve->dx());
      _bf->addTerm((D12_prev+D21_prev)*u2, ve->dx());
      _bf->addTerm((D21_prev+D12_prev)*u1, ve->dy());
      _bf->addTerm((D22_prev+D22_prev-2./3*(D11_prev+D22_prev))*u2, ve->dy());
      _bf->addTerm(u1_prev*(1*D11+1*D11-2./3*D11-2./3*D22), ve->dx());
      _bf->addTerm(u2_prev*(1*D12+1*D21), ve->dx());
      _bf->addTerm(u1_prev*(1*D21+1*D12), ve->dy());
      _bf->addTerm(u2_prev*(1*D22+1*D22-2./3*D11-2./3*D22), ve->dy());
      _bf->addTerm(te, ve);
      
      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev) * ve->dt());
      }
      if (_timeStepping)
      {
        _rhs->addTerm(Cv()*rho_prev_time*T_prev_time * ve);
        _rhs->addTerm(0.5*rho_prev_time*(u1_prev_time*u1_prev_time+u2_prev_time*u2_prev_time) * ve);
        _rhs->addTerm(-Cv()*rho_prev*T_prev * ve);
        _rhs->addTerm(-0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev) * ve);
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev * ve->dy());
      _rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(q2_prev * ve->dy());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev))*u1_prev * ve->dx());
      _rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
      _rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev))*u2_prev * ve->dy());
      break;
    case 3:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*rho, ve->dt());
        _bf->addTerm(-rho_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3), ve->dt());
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(Cv()*u2_prev*T_prev*rho+Cv()*rho_prev*T_prev*u2+Cv()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(Cv()*u3_prev*T_prev*rho+Cv()*rho_prev*T_prev*u3+Cv()*rho_prev*u3_prev*T), ve->dz());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2_prev*rho), ve->dy());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3_prev*rho), ve->dz());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2), ve->dy());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3), ve->dz());
      _bf->addTerm(-(rho_prev*u1_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dx());
      _bf->addTerm(-(rho_prev*u2_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dy());
      _bf->addTerm(-(rho_prev*u3_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dz());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u2), ve->dy());
      _bf->addTerm(-(R()*rho_prev*T_prev*u3), ve->dz());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*u2_prev*T_prev*rho), ve->dy());
      _bf->addTerm(-(R()*u3_prev*T_prev*rho), ve->dz());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(R()*rho_prev*u3_prev*T), ve->dz());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm(-q2, ve->dy());
      _bf->addTerm(-q3, ve->dz());
      _bf->addTerm((D11_prev+D11_prev-2./3*(D11_prev+D22_prev+D33_prev))*u1, ve->dx());
      _bf->addTerm((D12_prev+D21_prev)*u2, ve->dx());
      _bf->addTerm((D13_prev+D31_prev)*u3, ve->dx());
      _bf->addTerm((D21_prev+D12_prev)*u1, ve->dy());
      _bf->addTerm((D22_prev+D22_prev-2./3*(D11_prev+D22_prev+D33_prev))*u2, ve->dy());
      _bf->addTerm((D31_prev+D13_prev)*u3, ve->dy());
      _bf->addTerm((D31_prev+D13_prev)*u1, ve->dz());
      _bf->addTerm((D32_prev+D23_prev)*u2, ve->dz());
      _bf->addTerm((D33_prev+D33_prev-2./3*(D11_prev+D22_prev+D33_prev))*u3, ve->dz());
      _bf->addTerm(u1_prev*(D11+D11-2./3*D11-2./3*D22-2./3*D33), ve->dx());
      _bf->addTerm(u2_prev*(D12+D21), ve->dx());
      _bf->addTerm(u3_prev*(D13+D31), ve->dx());
      _bf->addTerm(u1_prev*(D21+D12), ve->dy());
      _bf->addTerm(u2_prev*(D22+D22-2./3*D11-2./3*D22-2./3*D33), ve->dy());
      _bf->addTerm(u3_prev*(D31+D13), ve->dy());
      _bf->addTerm(u1_prev*(D31+D13), ve->dz());
      _bf->addTerm(u2_prev*(D32+D23), ve->dz());
      _bf->addTerm(u3_prev*(D33+D33-2./3*D11-2./3*D22-2./3*D33), ve->dz());
      _bf->addTerm(te, ve);
      
      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(-0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev) * ve->dt());
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(Cv()*rho_prev*u3_prev*T_prev * ve->dz());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2_prev * ve->dy());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3_prev * ve->dz());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(R()*rho_prev*u3_prev*T_prev * ve->dz());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(q2_prev * ve->dy());
      _rhs->addTerm(q3_prev * ve->dz());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev+D33_prev))*u1_prev * ve->dx());
      _rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
      _rhs->addTerm(-(D13_prev+D31_prev)*u3_prev * ve->dx());
      _rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev+D33_prev))*u2_prev * ve->dy());
      _rhs->addTerm(-(D31_prev+D13_prev)*u3_prev * ve->dy());
      _rhs->addTerm(-(D31_prev+D13_prev)*u1_prev * ve->dz());
      _rhs->addTerm(-(D32_prev+D23_prev)*u2_prev * ve->dz());
      _rhs->addTerm(-(D33_prev+D33_prev-2./3*(D11_prev+D22_prev+D33_prev))*u3_prev * ve->dz());
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_spaceDim must be 1,2, or 3!");
  }
  
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
  
  LinearTermPtr adj_Cc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Ce = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fe = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD11 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD12 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD13 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD21 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD22 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD23 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD31 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD32 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD33 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD11 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD12 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD13 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD21 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD22 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD23 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD31 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD32 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD33 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Ge = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_vm = Teuchos::rcp( new LinearTerm );
  
  _ips["Graph"] = _bf->graphNorm();
  // cout << "Graph" << endl;
  // _ips["Graph"]->printInteractions();
  FunctionPtr rho_sqrt = Teuchos::rcp(new BoundedSqrtFunction(rho_prev,1e-4));
  FunctionPtr T_sqrt = Teuchos::rcp(new BoundedSqrtFunction(T_prev,1e-4));
  
  switch (_spaceDim)
  {
    case 1:
      adj_Cc->addTerm( vc->dt() + u1_prev*vm1->dt() + Cv()*T_prev*ve->dt() + 0.5*u1_prev*u1_prev*ve->dt() );
      adj_Cm1->addTerm( rho_prev*vm1->dt() + rho_prev*u1_prev*ve->dt() );
      adj_Ce->addTerm( Cv()*rho_prev*ve->dt() );
      adj_Fc->addTerm( u1_prev*vc->dx() + u1_prev*u1_prev*vm1->dx() + R()*T_prev*vm1->dx() + Cv()*T_prev*u1_prev*ve->dx()
                      + 0.5*u1_prev*u1_prev*u1_prev*ve->dx() + R()*T_prev*u1_prev*ve->dx() );
      adj_Fm1->addTerm( rho_prev*vc->dx() + 2*rho_prev*u1_prev*vm1->dx() + Cv()*T_prev*rho_prev*ve->dx()
                       + 0.5*rho_prev*u1_prev*u1_prev*ve->dx() + rho_prev*u1_prev*u1_prev*ve->dx() + R()*T_prev*rho_prev*ve->dx()
                       - D11_prev*ve->dx() - D11_prev*ve->dx() + 2./3*D11_prev*ve->dx() );
      adj_Fe->addTerm( R()*rho_prev*vm1->dx() + Cv()*rho_prev*u1_prev*ve->dx() + R()*rho_prev*u1_prev*ve->dx() );
      adj_KD11->addTerm( vm1->dx() + vm1->dx() - 2./3*vm1->dx() + u1_prev*ve->dx() + u1_prev*ve->dx() - 2./3*u1_prev*ve->dx() );
      adj_Kq1->addTerm( -ve->dx() );
      adj_MD11->addTerm( one*S1 );
      adj_Mq1->addTerm( Pr()/Cp()*tau );
      adj_Gm1->addTerm( one*S1->dx() );
      adj_Ge->addTerm( -tau->dx() );
      
      _ips["ManualGraph"] = Teuchos::rcp(new IP);
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_MD11 + adj_KD11 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_Mq1 + adj_Kq1 );
      if (_spaceTime)
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
      }
      else
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe );
      }
      _ips["ManualGraph"]->addTerm( vc );
      _ips["ManualGraph"]->addTerm( vm1 );
      _ips["ManualGraph"]->addTerm( ve );
      _ips["ManualGraph"]->addTerm( S1 );
      _ips["ManualGraph"]->addTerm( tau );
      
      _ips["EntropyGraph"] = Teuchos::rcp(new IP);
      _ips["EntropyGraph"]->addTerm( Cv()*T_sqrt/rho_sqrt*(1./_muFunc*adj_MD11 + adj_KD11) );
      _ips["EntropyGraph"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*(1./_muFunc*adj_Mq1 + adj_Kq1) );
      if (_spaceTime)
      {
        _ips["EntropyGraph"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Gc - adj_Fc - adj_Cc) );
        _ips["EntropyGraph"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Gm1 - adj_Fm1 - adj_Cm1) );
        _ips["EntropyGraph"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Ge - adj_Fe - adj_Ce) );
      }
      else
      {
        _ips["EntropyGraph"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Gc - adj_Fc) );
        _ips["EntropyGraph"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Gm1 - adj_Fm1) );
        _ips["EntropyGraph"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Ge - adj_Fe) );
      }
      _ips["EntropyGraph"]->addTerm( rho_sqrt/sqrt(_gamma-1)*vc );
      _ips["EntropyGraph"]->addTerm(    Cv()*T_sqrt/rho_sqrt*vm1 );
      _ips["EntropyGraph"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*ve );
      _ips["EntropyGraph"]->addTerm( Cv()*T_sqrt/rho_sqrt*S1 );
      _ips["EntropyGraph"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*tau );
      
      // cout << endl << "ManualGraph" << endl;
      // _ips["ManualGraph"]->printInteractions();
      
      _ips["Robust"] = Teuchos::rcp(new IP);
      // _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      // _ips["Robust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["Robust"]->addTerm( _muSqrtFunc*adj_KD11 );
      _ips["Robust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["Robust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["Robust"]->addTerm( adj_Fc + adj_Cc );
        _ips["Robust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["Robust"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["Robust"]->addTerm(_beta*v->grad());
        _ips["Robust"]->addTerm( adj_Fc );
        _ips["Robust"]->addTerm( adj_Fm1 );
        _ips["Robust"]->addTerm( adj_Fe );
      }
      // _ips["Robust"]->addTerm(tau->div());
      _ips["Robust"]->addTerm( adj_Gc );
      _ips["Robust"]->addTerm( adj_Gm1 );
      _ips["Robust"]->addTerm( adj_Ge );
      // _ips["Robust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["Robust"]->addTerm( vc );
      _ips["Robust"]->addTerm( vm1 );
      _ips["Robust"]->addTerm( ve );
      
      _ips["EntropyRobust"] = Teuchos::rcp(new IP);
      // _ips["EntropyRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["EntropyRobust"]->addTerm( Cv()*T_sqrt/rho_sqrt*Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["EntropyRobust"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      // _ips["EntropyRobust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["EntropyRobust"]->addTerm( Cv()*T_sqrt/rho_sqrt*_muSqrtFunc*adj_KD11 );
      _ips["EntropyRobust"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["EntropyRobust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["EntropyRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Fc + adj_Cc) );
        _ips["EntropyRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Fm1 + adj_Cm1) );
        _ips["EntropyRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Fe + adj_Ce) );
      }
      else
      {
        // _ips["EntropyRobust"]->addTerm(_beta*v->grad());
        _ips["EntropyRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Fc) );
        _ips["EntropyRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Fm1) );
        _ips["EntropyRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Fe) );
      }
      // _ips["EntropyRobust"]->addTerm(tau->div());
      _ips["EntropyRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*adj_Gc );
      _ips["EntropyRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*adj_Gm1 );
      _ips["EntropyRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*adj_Ge );
      // _ips["EntropyRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["EntropyRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["EntropyRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["EntropyRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["EntropyRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*vc );
      _ips["EntropyRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*vm1 );
      _ips["EntropyRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*ve );
      
      _ips["CoupledRobust"] = Teuchos::rcp(new IP);
      // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["CoupledRobust"]->addTerm( _muSqrtFunc*adj_KD11 );
      _ips["CoupledRobust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["CoupledRobust"]->addTerm( adj_Fc + adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fe );
      }
      // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["CoupledRobust"]->addTerm( vc );
      _ips["CoupledRobust"]->addTerm( vm1 );
      _ips["CoupledRobust"]->addTerm( ve );
      
      _ips["EntropyCoupledRobust"] = Teuchos::rcp(new IP);
      // _ips["EntropyCoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["EntropyCoupledRobust"]->addTerm( Cv()*T_sqrt/rho_sqrt*Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["EntropyCoupledRobust"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      // _ips["EntropyCoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["EntropyCoupledRobust"]->addTerm( Cv()*T_sqrt/rho_sqrt*_muSqrtFunc*adj_KD11 );
      _ips["EntropyCoupledRobust"]->addTerm(      T_sqrt*T_sqrt/rho_sqrt*sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["EntropyCoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
        _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Gc - adj_Fc - adj_Cc) );
        _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Gm1 - adj_Fm1 - adj_Cm1) );
        _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Ge - adj_Fe - adj_Ce) );
        // _ips["EntropyCoupledRobust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Fc + adj_Cc) );
        _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Fm1 + adj_Cm1) );
        _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Fe + adj_Ce) );
      }
      else
      {
        // _ips["EntropyCoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*(adj_Gc - adj_Fc) );
        _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*(adj_Gm1 - adj_Fm1) );
        _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*(adj_Ge - adj_Fe) );
        // _ips["EntropyCoupledRobust"]->addTerm(_beta*v->grad());
        _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*adj_Fc );
        _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*adj_Fm1 );
        _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*adj_Fe );
      }
      // _ips["EntropyCoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["EntropyCoupledRobust"]->addTerm( rho_sqrt/sqrt(_gamma-1)*vc );
      _ips["EntropyCoupledRobust"]->addTerm(    Cv()*T_sqrt/rho_sqrt*vm1 );
      _ips["EntropyCoupledRobust"]->addTerm(         T_sqrt*T_sqrt/rho_sqrt*ve );
      
      _ips["NSDecoupled"] = Teuchos::rcp(new IP);
      // _ips["NSDecoupled"]->addTerm(one/Function::h()*tau);
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_MD11 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_Mq1 );
      // _ips["NSDecoupled"]->addTerm(tau->div());
      _ips["NSDecoupled"]->addTerm( adj_KD11 );
      _ips["NSDecoupled"]->addTerm( adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["NSDecoupled"]->addTerm(_beta*v->grad() + v->dt());
        _ips["NSDecoupled"]->addTerm( adj_Fc + adj_Cc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["NSDecoupled"]->addTerm(_beta*v->grad());
        _ips["NSDecoupled"]->addTerm( adj_Fc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fe );
      }
      if (_timeStepping)
      {
        // _ips["NSDecoupled"]->addTerm(_beta*v->grad() + v->dt());
        _ips["NSDecoupled"]->addTerm( 1./_dt*vc + u1_prev/_dt*vm1 + Cv()*T_prev/_dt*ve + 0.5*u1_prev*u1_prev/_dt*ve );
        _ips["NSDecoupled"]->addTerm( rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve );
        _ips["NSDecoupled"]->addTerm( Cv()*rho_prev/_dt*ve );
      }
      // _ips["NSDecoupled"]->addTerm(v->grad());
      _ips["NSDecoupled"]->addTerm( adj_Gc );
      _ips["NSDecoupled"]->addTerm( adj_Gm1 );
      _ips["NSDecoupled"]->addTerm( adj_Ge );
      // _ips["NSDecoupled"]->addTerm(v);
      _ips["NSDecoupled"]->addTerm( vc );
      _ips["NSDecoupled"]->addTerm( vm1 );
      _ips["NSDecoupled"]->addTerm( ve );
      break;
    case 2:
      adj_Cc->addTerm( vc->dt() + u1_prev*vm1->dt() + u2_prev*vm2->dt() + Cv()*T_prev*ve->dt() + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*ve->dt() );
      adj_Cm1->addTerm( rho_prev*vm1->dt() + rho_prev*u1_prev*ve->dt() );
      adj_Cm2->addTerm( rho_prev*vm2->dt() + rho_prev*u2_prev*ve->dt() );
      adj_Ce->addTerm( Cv()*rho_prev*ve->dt() );
      adj_Fc->addTerm( u1_prev*vc->dx() + u2_prev*vc->dy()
                      + u1_prev*u1_prev*vm1->dx() + u1_prev*u2_prev*vm1->dy() + u2_prev*u1_prev*vm2->dx() + u2_prev*u2_prev*vm2->dy()
                      + R()*T_prev*vm1->dx() + R()*T_prev*vm2->dy()
                      + Cv()*T_prev*u1_prev*ve->dx() + Cv()*T_prev*u2_prev*ve->dy()
                      + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*(u1_prev*ve->dx() + u2_prev*ve->dy())
                      + R()*T_prev*u1_prev*ve->dx() + R()*T_prev*u2_prev*ve->dy() );
      adj_Fm1->addTerm( rho_prev*vc->dx()
                       + 2*rho_prev*u1_prev*vm1->dx() + rho_prev*u2_prev*vm1->dy() + rho_prev*u2_prev*vm2->dx()
                       + Cv()*T_prev*rho_prev*ve->dx()
                       + 0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*ve->dx()
                       + rho_prev*u1_prev*(u1_prev*ve->dx() + u2_prev*ve->dy()) + R()*T_prev*rho_prev*ve->dx()
                       - 2*D11_prev*ve->dx() - D12_prev*ve->dy() - D21_prev*ve->dy()
                       + 2./3*(D11_prev + D22_prev)*ve->dx() );
      adj_Fm2->addTerm( rho_prev*vc->dy()
                       + rho_prev*u1_prev*vm1->dy() + rho_prev*u1_prev*vm2->dx()+ 2*rho_prev*u2_prev*vm2->dy()
                       + Cv()*T_prev*rho_prev*ve->dy()
                       + 0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*ve->dy()
                       + rho_prev*u2_prev*(u1_prev*ve->dx() + u2_prev*ve->dy()) + R()*T_prev*rho_prev*ve->dy()
                       - D21_prev*ve->dx() - D12_prev*ve->dx() - 2*D22_prev*ve->dy()
                       + 2./3*(D11_prev + D22_prev)*ve->dy() );
      adj_Fe->addTerm( R()*rho_prev*(vm1->dx() + vm2->dy()) + Cv()*rho_prev*(u1_prev*ve->dx()+u2_prev*ve->dy())
                      + R()*rho_prev*(u1_prev*ve->dx()+u2_prev*ve->dy()) );
      adj_KD11->addTerm( vm1->dx() + vm1->dx() - 2./3*vm1->dx() - 2./3*vm2->dy()
                        + u1_prev*ve->dx() + u1_prev*ve->dx() - 2./3*u1_prev*ve->dx() - 2./3*u2_prev*ve->dy() );
      adj_KD12->addTerm( vm1->dy() + vm2->dx() + u1_prev*ve->dy() + u2_prev*ve->dx() );
      adj_KD21->addTerm( vm2->dx() + vm1->dy() + u2_prev*ve->dx() + u1_prev*ve->dy() );
      adj_KD22->addTerm( vm2->dy() + vm2->dy() - 2./3*vm1->dx() - 2./3*vm2->dy()
                        + u2_prev*ve->dy() + u2_prev*ve->dy() - 2./3*u1_prev*ve->dx() - 2./3*u2_prev*ve->dy() );
      adj_Kq1->addTerm( -ve->dx() );
      adj_Kq2->addTerm( -ve->dy() );
      adj_MD11->addTerm( one*S1->x() );
      adj_MD12->addTerm( one*S1->y() );
      adj_MD21->addTerm( one*S2->x() );
      adj_MD22->addTerm( one*S2->y() );
      adj_Mq1->addTerm( Pr()/Cp()*tau->x() );
      adj_Mq2->addTerm( Pr()/Cp()*tau->y() );
      adj_Gm1->addTerm( one*S1->div() );
      adj_Gm2->addTerm( one*S2->div() );
      adj_Ge->addTerm( -tau->div() );
      
      _ips["ManualGraph"] = Teuchos::rcp(new IP);
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_MD11 + adj_KD11 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_MD12 + adj_KD12 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_MD21 + adj_KD21 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_MD22 + adj_KD22 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_Mq1 + adj_Kq1 );
      _ips["ManualGraph"]->addTerm( 1./_muFunc*adj_Mq2 + adj_Kq2 );
      if (_spaceTime)
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["ManualGraph"]->addTerm( adj_Gm2 - adj_Fm2 - adj_Cm2 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
      }
      else
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["ManualGraph"]->addTerm( adj_Gm2 - adj_Fm2 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe );
      }
      _ips["ManualGraph"]->addTerm( vc );
      _ips["ManualGraph"]->addTerm( vm1 );
      _ips["ManualGraph"]->addTerm( vm2 );
      _ips["ManualGraph"]->addTerm( ve );
      _ips["ManualGraph"]->addTerm( S1);
      _ips["ManualGraph"]->addTerm( S2 );
      _ips["ManualGraph"]->addTerm( tau );
      
      _ips["Robust"] = Teuchos::rcp(new IP);
      // _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD12);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD21);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD22);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq2);
      // _ips["Robust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["Robust"]->addTerm( _muSqrtFunc*adj_KD11 );
      _ips["Robust"]->addTerm( _muSqrtFunc*adj_KD12 );
      _ips["Robust"]->addTerm( _muSqrtFunc*adj_KD21 );
      _ips["Robust"]->addTerm( _muSqrtFunc*adj_KD22 );
      _ips["Robust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      _ips["Robust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq2 );
      if (_spaceTime)
      {
        // _ips["Robust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["Robust"]->addTerm( adj_Fc + adj_Cc );
        _ips["Robust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["Robust"]->addTerm( adj_Fm2 + adj_Cm2 );
        _ips["Robust"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["Robust"]->addTerm(_beta*v->grad());
        _ips["Robust"]->addTerm( adj_Fc );
        _ips["Robust"]->addTerm( adj_Fm1 );
        _ips["Robust"]->addTerm( adj_Fm2 );
        _ips["Robust"]->addTerm( adj_Fe );
      }
      if (_timeStepping)
      {
        _ips["Robust"]->addTerm( 1./_dt*vc + u1_prev/_dt*vm1 + u2_prev/_dt*vm2 + Cv()*T_prev/_dt*ve
                                + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)/_dt*ve );
        _ips["Robust"]->addTerm( rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve );
        _ips["Robust"]->addTerm( rho_prev/_dt*vm2 + rho_prev*u2_prev/_dt*ve );
        _ips["Robust"]->addTerm( Cv()*rho_prev/_dt*ve );
      }
      // _ips["Robust"]->addTerm(tau->div());
      _ips["Robust"]->addTerm( adj_Gc );
      _ips["Robust"]->addTerm( adj_Gm1 );
      _ips["Robust"]->addTerm( adj_Gm2 );
      _ips["Robust"]->addTerm( adj_Ge );
      // _ips["Robust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm2 );
      // _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["Robust"]->addTerm( vc );
      _ips["Robust"]->addTerm( vm1 );
      _ips["Robust"]->addTerm( vm2 );
      _ips["Robust"]->addTerm( ve );
      
      _ips["CoupledRobust"] = Teuchos::rcp(new IP);
      // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD11);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD12);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD21);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_MD22);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq1);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),1./_muSqrtFunc)*adj_Mq2);
      // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["CoupledRobust"]->addTerm( _muSqrtFunc*adj_KD11 );
      _ips["CoupledRobust"]->addTerm( _muSqrtFunc*adj_KD12 );
      _ips["CoupledRobust"]->addTerm( _muSqrtFunc*adj_KD21 );
      _ips["CoupledRobust"]->addTerm( _muSqrtFunc*adj_KD22 );
      _ips["CoupledRobust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq1 );
      _ips["CoupledRobust"]->addTerm( sqrt(Cp()/Pr())*_muSqrtFunc*adj_Kq2 );
      if (_spaceTime)
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Gm2 - adj_Fm2 - adj_Cm2 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["CoupledRobust"]->addTerm( adj_Fc + adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fm2 + adj_Cm2 );
        _ips["CoupledRobust"]->addTerm( adj_Fe + adj_Ce );
      }
      else if (_timeStepping)
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Gm2 - adj_Fm2 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fm2 );
        _ips["CoupledRobust"]->addTerm( adj_Fe );
        _ips["CoupledRobust"]->addTerm( 1./_dt*vc + u1_prev/_dt*vm1 + u2_prev/_dt*vm2 + Cv()*T_prev/_dt*ve
                                       + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)/_dt*ve );
        _ips["CoupledRobust"]->addTerm( rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve );
        _ips["CoupledRobust"]->addTerm( rho_prev/_dt*vm2 + rho_prev*u2_prev/_dt*ve );
        _ips["CoupledRobust"]->addTerm( Cv()*rho_prev/_dt*ve );
        
        // // _ips["CoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        // _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc + 1./_dt*vc + u1_prev/_dt*vm1 + u2_prev/_dt*vm2 + Cv()*T_prev/_dt*ve
        //     + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)/_dt*ve );
        // _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 + rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve);
        // _ips["CoupledRobust"]->addTerm( adj_Gm2 - adj_Fm2 + rho_prev/_dt*vm2 + rho_prev*u2_prev/_dt*ve);
        // _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe + Cv()*rho_prev/_dt*ve);
        // // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
        // _ips["CoupledRobust"]->addTerm( adj_Fc );
        // _ips["CoupledRobust"]->addTerm( adj_Fm1 );
        // _ips["CoupledRobust"]->addTerm( adj_Fm2 );
        // _ips["CoupledRobust"]->addTerm( adj_Fe );
      }
      else
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Gm2 - adj_Fm2 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fm2 );
        _ips["CoupledRobust"]->addTerm( adj_Fe );
      }
      // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm2 );
      // _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );
      _ips["CoupledRobust"]->addTerm( vc );
      _ips["CoupledRobust"]->addTerm( vm1 );
      _ips["CoupledRobust"]->addTerm( vm2 );
      _ips["CoupledRobust"]->addTerm( ve );
      
      _ips["NSDecoupled"] = Teuchos::rcp(new IP);
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_MD11 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_MD12 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_MD21 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_MD22 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_Mq1 );
      _ips["NSDecoupled"]->addTerm( 1./Function::h()*adj_Mq2 );
      _ips["NSDecoupled"]->addTerm( adj_KD11 );
      _ips["NSDecoupled"]->addTerm( adj_KD12 );
      _ips["NSDecoupled"]->addTerm( adj_KD21 );
      _ips["NSDecoupled"]->addTerm( adj_KD22 );
      _ips["NSDecoupled"]->addTerm( adj_Kq1 );
      _ips["NSDecoupled"]->addTerm( adj_Kq2 );
      if (_spaceTime)
      {
        _ips["NSDecoupled"]->addTerm( adj_Fc + adj_Cc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fm2 + adj_Cm2 );
        _ips["NSDecoupled"]->addTerm( adj_Fe + adj_Ce );
      }
      else if (_timeStepping)
      {
        // _ips["NSDecoupled"]->addTerm( 1./_dt*vc + u1_prev/_dt*vm1 + u2_prev/_dt*vm2 + Cv()*T_prev/_dt*ve
        //     + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)/_dt*ve );
        // _ips["NSDecoupled"]->addTerm( rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve );
        // _ips["NSDecoupled"]->addTerm( rho_prev/_dt*vm2 + rho_prev*u2_prev/_dt*ve );
        // _ips["NSDecoupled"]->addTerm( Cv()*rho_prev/_dt*ve );
        
        // _ips["CoupledRobust"]->addTerm(beta*v->grad());
        _ips["NSDecoupled"]->addTerm( adj_Fc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fm2 );
        _ips["NSDecoupled"]->addTerm( adj_Fe );
        _ips["NSDecoupled"]->addTerm( 1./_dt*vc + u1_prev/_dt*vm1 + u2_prev/_dt*vm2 + Cv()*T_prev/_dt*ve
                                     + 0.5*(u1_prev*u1_prev+u2_prev*u2_prev)/_dt*ve );
        _ips["NSDecoupled"]->addTerm( rho_prev/_dt*vm1 + rho_prev*u1_prev/_dt*ve);
        _ips["NSDecoupled"]->addTerm( rho_prev/_dt*vm2 + rho_prev*u2_prev/_dt*ve);
        _ips["NSDecoupled"]->addTerm( Cv()*rho_prev/_dt*ve);
      }
      else
      {
        _ips["NSDecoupled"]->addTerm( adj_Fc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fm2 );
        _ips["NSDecoupled"]->addTerm( adj_Fe );
      }
      _ips["NSDecoupled"]->addTerm( adj_Gc );
      _ips["NSDecoupled"]->addTerm( adj_Gm1 );
      _ips["NSDecoupled"]->addTerm( adj_Gm2 );
      _ips["NSDecoupled"]->addTerm( adj_Ge );
      _ips["NSDecoupled"]->addTerm( vc );
      _ips["NSDecoupled"]->addTerm( vm1 );
      _ips["NSDecoupled"]->addTerm( vm2 );
      _ips["NSDecoupled"]->addTerm( ve );
      break;
    case 3:
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_spaceDim must be 1,2, or 3!");
  }
  
  IPPtr ip = _ips.at(normName);
  
  // set the inner product to the graph norm:
  // setIP( _ips[normName] );
  
  // this->setForcingFunction(Teuchos::null); // will default to zero
  // _rhsForSolve = this->rhs(_neglectFluxesOnRHS);
  // _rhsForResidual = this->rhs(false);
  // _solnIncrement->setRHS(_rhsForSolve);
  
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
