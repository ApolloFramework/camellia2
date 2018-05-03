//
//  IdealMHDFormulation.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 5/1/18.
//

#include "IdealMHDFormulation.hpp"

#include "BC.h"
#include "BF.h"
#include "ExpFunction.h"
#include "Functions.hpp"
#include "LagrangeConstraints.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"
#include "VarFunction.h"

using namespace Camellia;
using namespace std;

const string IdealMHDFormulation::S_rho = "rho";
const string IdealMHDFormulation::S_m1  = "m1";
const string IdealMHDFormulation::S_m2  = "m2";
const string IdealMHDFormulation::S_m3  = "m3";
const string IdealMHDFormulation::S_E   = "E";
const string IdealMHDFormulation::S_B1 = "B1";
const string IdealMHDFormulation::S_B2 = "B2";
const string IdealMHDFormulation::S_B3 = "B3";

const string IdealMHDFormulation::S_tc = "tc";
const string IdealMHDFormulation::S_tm1 = "tm1";
const string IdealMHDFormulation::S_tm2 = "tm2";
const string IdealMHDFormulation::S_tm3 = "tm3";
const string IdealMHDFormulation::S_te = "te";
const string IdealMHDFormulation::S_tB1 = "tB1";
const string IdealMHDFormulation::S_tB2 = "tB2";
const string IdealMHDFormulation::S_tB3 = "tB3";

const string IdealMHDFormulation::S_vc = "vc";
const string IdealMHDFormulation::S_vm1  = "vm1";
const string IdealMHDFormulation::S_vm2  = "vm2";
const string IdealMHDFormulation::S_vm3  = "vm3";
const string IdealMHDFormulation::S_ve   = "ve";
const string IdealMHDFormulation::S_vB1  = "vB1";
const string IdealMHDFormulation::S_vB2  = "vB2";
const string IdealMHDFormulation::S_vB3  = "vB3";

const string IdealMHDFormulation::S_m[3]    = {S_m1, S_m2, S_m3};
const string IdealMHDFormulation::S_B[3]    = {S_B1, S_B2, S_B3};
const string IdealMHDFormulation::S_tm[3]   = {S_tm1, S_tm2, S_tm3};
const string IdealMHDFormulation::S_tB[3]   = {S_tB1, S_tB2, S_tB3};
const string IdealMHDFormulation::S_vm[3]   = {S_vm1, S_vm2, S_vm3};
const string IdealMHDFormulation::S_vB[3]   = {S_vB1, S_vB2, S_vB3};

void IdealMHDFormulation::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  const int trueSpaceDim = 3;
  if ((i > trueSpaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::steadyFormulation(int spaceDim, MeshTopologyPtr meshTopo, int polyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("spatialPolyOrder", polyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::timeSteppingFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::steadyEulerFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::timeSteppingEulerFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

IdealMHDFormulation::IdealMHDFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _spaceDim = spaceDim;
  const int trueSpaceDim = 3; // as opposed to _spaceDim, which indicates the number of dimensions that can have non-zero derivatives (also the dimension of the mesh)

  _fc = ParameterFunction::parameterFunction(0.0);
  _fe = ParameterFunction::parameterFunction(0.0);
  _fm = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim, ParameterFunction::parameterFunction(0.0));
  _fB = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim, ParameterFunction::parameterFunction(0.0));
  _fc->setName("fc");
  _fe->setName("fe");
  for (int d=0; d<spaceDim; d++)
  {
    ostringstream name_fm;
    name_fm << "fm" << d+1;
    _fm[d]->setName(name_fm.str());
    ostringstream name_fB;
    name_fm << "fB" << d+1;
    _fB[d]->setName(name_fB.str());
  }
  _gamma = parameters.get<double>("gamma",1.4);
  _Pr = parameters.get<double>("Pr",0.713);
  _Cv = parameters.get<double>("Cv",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string normName = parameters.get<string>("norm", "Graph");
  
  // nonlinear parameters
  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double initialDt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  
  string problemName = parameters.get<string>("problemName", "");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");
  
  _useConformingTraces = useConformingTraces;
  _spatialPolyOrder = spatialPolyOrder;
  _temporalPolyOrder =temporalPolyOrder;
  _dt = ParameterFunction::parameterFunction(initialDt);
  _dt->setName("dt");
  _t = ParameterFunction::parameterFunction(0);
  _t0 = parameters.get<double>("t0",0);
  _delta_k = delta_k;
  
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;
  
  
  // TEUCHOS_TEST_FOR_EXCEPTION(_timeStepping, std::invalid_argument, "Time stepping not supported");
  
  // field variables
  VarPtr rhoVar;
  vector<VarPtr> mVar(trueSpaceDim);
  VarPtr EVar;
  vector<VarPtr> BVar(trueSpaceDim); // In 1D, first component will be null
  FunctionPtr B; // 3-component vector Function.  Some or all components are "abstract" functions (i.e., dependent on Vars)
  
  // trace variables
  VarPtr tc;
  vector<VarPtr> tm(trueSpaceDim);
  VarPtr te;
  vector<VarPtr> tB(trueSpaceDim); // first component null in 1D
  
  
  // test variables
  VarPtr vc;
  vector<VarPtr> vm(trueSpaceDim);
  VarPtr ve;
  vector<VarPtr> vB(trueSpaceDim);
  
  _vf = VarFactory::varFactory();
  
  rhoVar = _vf->fieldVar(S_rho);
  
  for (int d=0; d<trueSpaceDim; d++)
  {
    mVar[d] = _vf->fieldVar(S_m[d]);
  }
  
  if (_spaceDim > 1)
  {
    BVar[0] = _vf->fieldVar(S_B[0]);
  }
  for (int d=1; d<trueSpaceDim; d++)
  {
    BVar[d] = _vf->fieldVar(S_B[d]);
  }
  
  vector<FunctionPtr> B_comps(trueSpaceDim);
  if (_spaceDim == 1)
  {
    // then the vars that remain as unknowns are By, Bz
    B_comps[0] = Function::constant(0.75); // TODO: make this parameter-controlled.  (This is the value we use for Brio-Wu.)
    B_comps[1] = VarFunction<double>::abstractFunction(BVar[1]);
    B_comps[2] = VarFunction<double>::abstractFunction(BVar[2]);
  }
  else
  {
    B_comps[0] = VarFunction<double>::abstractFunction(BVar[0]);
    B_comps[1] = VarFunction<double>::abstractFunction(BVar[1]);
    B_comps[2] = VarFunction<double>::abstractFunction(BVar[2]);
  }
  B = Function::vectorize(B_comps);
  
  EVar = _vf->fieldVar(S_E);
  
  // FunctionPtr n = Function::normal();
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  // TODO: add in here the definitions of the LinearTerms that the fluxes below trace.
  //       (See the exactSolution_tc(), etc. methods for an indication of the right thing here.)
  tc = _vf->fluxVar(S_tc);
  for (int d=0; d<trueSpaceDim; d++)
  {
    tm[d] = _vf->fluxVar(S_tm[d]);
  }
  te = _vf->fluxVar(S_te);
  
  vc = _vf->testVar(S_vc, HGRAD);
  for (int d=0; d<trueSpaceDim; d++)
  {
    vm[d] = _vf->testVar(S_vm[d], HGRAD);
  }
  ve = _vf->testVar(S_ve, HGRAD);
  
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    vB[d] = _vf->testVar(S_vB[d], HGRAD);
    tB[d] = _vf->fluxVar(S_tB[d]);
  }
  
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
  
  // Let's try writing the flux terms in terms of Functions
  // We write these in terms of u, m, P*, B, E
  FunctionPtr massFlux, momentumFlux, magneticFlux, energyFlux;
  {
    int trueSpaceDim = 3; // as opposed to our mesh space dim, _spaceDim.
    vector<FunctionPtr> m_comps;
    for (int d=0; d<trueSpaceDim; d++)
    {
      auto m_comp = VarFunction<double>::abstractFunction(mVar[d]);
      m_comps.push_back(m_comp);
    }
    
    FunctionPtr E   = VarFunction<double>::abstractFunction(EVar);
    FunctionPtr m   = Function::vectorize(m_comps);
    FunctionPtr rho = VarFunction<double>::abstractFunction(rhoVar);
    FunctionPtr u   = m / rho;
    // (gas) pressure is (gamma - 1) * (E - 0.5 * m * u)
    FunctionPtr p   = (gamma() - 1.0) * (E - 0.5 * dot(trueSpaceDim,m,u));
    // P* is p + 0.5 * (B * B)
//    cout << "B: " << B->displayString() << endl;
    FunctionPtr p_star = p + 0.5 * dot(trueSpaceDim, B, B);
    FunctionPtr I   = identityMatrix<double>(trueSpaceDim);
    
    massFlux     = m;
    momentumFlux = outerProduct(trueSpaceDim, u, m) + p_star * I - outerProduct(trueSpaceDim, B, B);
    energyFlux   = (E + p_star) * u - B * dot(trueSpaceDim,B,u);
    magneticFlux = outerProduct(trueSpaceDim, u, B) - outerProduct(trueSpaceDim, B, u);
    
    if (_spaceDim == 1)
    {
      // take the x component of each flux term: this is the 1D flux (we drop y and z derivative terms)
      int x_comp = 1;
      massFlux     = massFlux->spatialComponent(x_comp);
      momentumFlux = momentumFlux->spatialComponent(x_comp);
      energyFlux   = energyFlux->spatialComponent(x_comp);
      magneticFlux = magneticFlux->spatialComponent(x_comp); // the first component of this is zero; we don't test against this in 1D
    }
    else if (_spaceDim == 2)
    {
      // take x,y components of each flux: this is 2D flux (we drop z derivative terms)
      int x_comp = 1, y_comp = 2;
      massFlux     = Function::vectorize(massFlux->spatialComponent    (x_comp),  massFlux->spatialComponent    (y_comp));
      momentumFlux = Function::vectorize(momentumFlux->spatialComponent(x_comp),  momentumFlux->spatialComponent(y_comp));
      energyFlux   = Function::vectorize(energyFlux->spatialComponent  (x_comp),  energyFlux->spatialComponent  (y_comp));
      magneticFlux = Function::vectorize(magneticFlux->spatialComponent(x_comp),  magneticFlux->spatialComponent(y_comp));
    }
    
//    cout << "massFlux: " << massFlux->displayString() << endl;
//    cout << "energyFlux: " << energyFlux->displayString() << endl;
//    cout << "momentumFlux: " << momentumFlux->displayString() << endl;
//    cout << "magneticFlux: " << magneticFlux->displayString() << endl;
  }
  
  _bf = Teuchos::rcp( new BF(_vf) );
  _steadyBF = Teuchos::rcp( new BF(_vf) );
  BFPtr transientBF = Teuchos::rcp( new BF(_vf) ); // we'll end up writing _bf = _steadyBF + transientBF, basically
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
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
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
  
  _fluxEquations[vc->ID()] = {vc,massFlux,tc,rhoVar,_fc};
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto momentumColumn = (_spaceDim > 1) ? column(_spaceDim,momentumFlux,d+1) : momentumFlux->spatialComponent(d+1);
    _fluxEquations[vm[d]->ID()] = {vm[d],momentumColumn,tm[d],mVar[d],_fm[d]};
    if ((d > 0) || (_spaceDim != 1))
    {
      auto magneticColumn = (_spaceDim > 1) ? column(_spaceDim,magneticFlux,d+1) : magneticFlux->spatialComponent(d+1);
      _fluxEquations[vB[d]->ID()] = {vB[d],magneticColumn,tB[d],BVar[d],_fB[d]};
    }
  }
  _fluxEquations[ve->ID()] = {ve,energyFlux,te,EVar,_fe};
  
  std::string backgroundFlowIdentifierExponent = "";
  std::string previousTimeIdentifierExponent = "";
  if (_timeStepping)
  {
    backgroundFlowIdentifierExponent = "k";
    previousTimeIdentifierExponent = "k-1";
  }
  // to avoid needing a bunch of casts below, do a cast once here:
  FunctionPtr dt = (FunctionPtr)_dt;
  
  std::map<int,FunctionPtr> backgroundFlowFunctions, previousTimeFunctions;
  bool weightFluxesByParity = true; // inconsequential for present purposes; will want to revisit if we store these maps as member variables...
  for (auto varEntry : trialVars)
  {
    VarPtr var = varEntry.second;
    backgroundFlowFunctions[var->ID()] = Function::solution(var, _backgroundFlow, weightFluxesByParity, backgroundFlowIdentifierExponent);
    previousTimeFunctions[var->ID()]   = Function::solution(var, _solnPrevTime,   weightFluxesByParity, previousTimeIdentifierExponent);
  }
  
  for (auto eqnEntry : _fluxEquations)
  {
    auto eqn = eqnEntry.second;
    auto testVar  = eqn.testVar;
    auto timeTerm = eqn.timeTerm;
    auto flux     = eqn.flux;
    auto traceVar = eqn.traceVar;
    auto f_rhs    = eqn.f_rhs;
    auto timeTerm_prev      = backgroundFlowFunctions[timeTerm->ID()];
    auto timeTerm_prev_time = previousTimeFunctions  [timeTerm->ID()];
    if (_spaceTime)
    {
      _bf ->addTerm(-timeTerm,      testVar->dt());
      _rhs->addTerm(timeTerm_prev * testVar->dt());
    }
    else if (_timeStepping)
    {
      _bf ->addTerm(  timeTerm / dt, testVar);
      _rhs->addTerm(- (timeTerm_prev - timeTerm_prev_time) / dt  * testVar);
    }
//    cout << "Test Var: " << testVar->displayString() << endl;
//    cout << "Flux:     " << flux->displayString() << endl;
    auto fluxJacobian = flux->jacobian(backgroundFlowFunctions);
    auto fluxPrevious = flux->evaluateAt(backgroundFlowFunctions);
    Camellia::EOperator gradOp = (_spaceDim > 1) ? OP_GRAD : OP_DX;
    
    _bf->      addTerm(-fluxJacobian, testVar->applyOp(gradOp)); // negative from integration by parts
    _steadyBF->addTerm(-fluxJacobian, testVar->applyOp(gradOp));
    _rhs     ->addTerm(fluxPrevious * testVar->applyOp(gradOp));
    
    _bf->      addTerm(traceVar, testVar);
    _steadyBF->addTerm(traceVar, testVar);
    _rhs     ->addTerm(f_rhs * testVar);
  }
  
  vector<VarPtr> missingTestVars = _bf->missingTestVars();
  vector<VarPtr> missingTrialVars = _bf->missingTrialVars();
  if (missingTestVars.size() > 0)
    cout << "WARNING: in IdealMHDFormulation, missing test vars:\n";
  for (auto var : missingTestVars)
  {
    cout << var->displayString() << endl;
  }
  if (missingTrialVars.size() > 0)
    cout << "WARNING: in IdealMHDFormulation, missing trial vars:\n";
  for (auto var : missingTrialVars)
  {
    cout << var->displayString() << endl;
  }

  IPPtr ip = _bf->graphNorm();
  
  _solnIncrement->setRHS(_rhs);
  _solnIncrement->setIP(ip);
  
  mesh->registerSolution(_backgroundFlow);
  mesh->registerSolution(_solnIncrement);
  mesh->registerSolution(_solnPrevTime);
  
  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );
  
  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );
  
  // Set up Functions for L^2 norm computations
  auto fieldVars = _vf->fieldVars();
  _L2IncrementFunction = Function::zero();
  _L2SolutionFunction = Function::zero();
  for (auto fieldVar : fieldVars)
  {
    auto fieldIncrement = Function::solution(fieldVar, _solnIncrement);
    _L2IncrementFunction = _L2IncrementFunction + fieldIncrement * fieldIncrement;
    auto fieldSolution = Function::solution(fieldVar, _backgroundFlow);
    _L2SolutionFunction = _L2SolutionFunction + fieldSolution * fieldSolution;
  }
  
  _solver = Solver::getDirectSolver();
  
  _nonlinearIterationCount = 0;
}

void IdealMHDFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr tc_exact)
{
  VarPtr tc = this->tc();
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
}

void IdealMHDFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr te_exact)
{
  VarPtr te = this->te();
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
}

void IdealMHDFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  VarPtr tc = this->tc();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tc_exact = this->exactSolution_tc(u, rho, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
}

void IdealMHDFormulation::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tm_exact = exactSolution_tm(u, rho, E, B, includeParity);
  for (int d=0; d<_spaceDim; d++)
  {
    VarPtr tm_i = this->tm(d+1);
    _solnIncrement->bc()->addDirichlet(tm_i, region, tm_exact[d]);
  }
}

void IdealMHDFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  VarPtr te = this->te();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto te_exact = exactSolution_te(u, rho, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
}

VarPtr IdealMHDFormulation::B(int i)
{
  return _vf->trialVar(S_B[i]);
}

BFPtr IdealMHDFormulation::bf()
{
  return _bf;
}

double IdealMHDFormulation::Cv()
{
  return _Cv;
}

double IdealMHDFormulation::Cp()
{
  return _gamma*_Cv;
}

VarPtr IdealMHDFormulation::E()
{
  return _vf->fieldVar(S_E);
}

// ! For an exact solution (u, rho, E, B), returns the corresponding forcing in the continuity equation
FunctionPtr IdealMHDFormulation::exactSolution_fc(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _massFlux;
  FunctionPtr timeTerm = rho; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(u, rho, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);

  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
FunctionPtr IdealMHDFormulation::exactSolution_fe(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _energyFlux;
  FunctionPtr timeTerm = E; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(u, rho, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);
  
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_fm(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _momentumFlux;
  
  vector<FunctionPtr> f(3);

  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(u, rho, E, B, includeFluxParity);
  auto fluxes = abstractFlux->evaluateAt(exactMap);
  
  for (int i=0; i<3; i++)
  {
    // get the column (row in 1D) of the momentum tensor
    auto flux = (_spaceDim > 1) ? column(_spaceDim,fluxes,i+1) : fluxes->spatialComponent(i+1);
    
    FunctionPtr timeTerm = (rho * u)->spatialComponent(i+1); // the thing that gets differentiated in time
    f[i] = Function::zero();
    if (timeTerm->dt() != Teuchos::null)
    {
      f[i] = f[i] + timeTerm->dt();
    }
    
    Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    f[i] = f[i] + Function::op(flux, divOp);
  }
  return f;
}

FunctionPtr IdealMHDFormulation::exactSolution_tc(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->vc();
  return exactSolutionFlux(testVar, u, rho, E, B, includeParity);
}

FunctionPtr IdealMHDFormulation::exactSolution_te(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->ve();
  return exactSolutionFlux(testVar, u, rho, E, B, includeParity);
}

std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_tm(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  const int trueSpaceDim = 3;
  std::vector<FunctionPtr> tm(trueSpaceDim);
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto testVar = this->vm(d+1);
    tm[d] = exactSolutionFlux(testVar, u, rho, E, B, includeParity);
  }
  return tm;
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionFieldMap(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B)
{
  map<int, FunctionPtr> exactMap;
  const int trueSpaceDim = 3;
  exactMap[this->rho()->ID()] = rho;
  exactMap[this->E()->ID()] = E;
  for (int d=1; d<trueSpaceDim; d++)
  {
    exactMap[this->m(d)->ID()] = rho * u->spatialComponent(d);
    if ((d > 1) || (_spaceDim > 1))
    {
      exactMap[this->B(d)->ID()] = B->spatialComponent(d);
    }
  }
  return exactMap;
}

FunctionPtr IdealMHDFormulation::exactSolutionFlux(VarPtr testVar, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B,
                                                   bool includeParity)
{
  ScalarFluxEquation equation = _fluxEquations.find(testVar->ID())->second;
  auto abstractFlux = equation.flux;
  auto timeTerm = equation.timeTerm;
  
  map<int,FunctionPtr> exactMap = exactSolutionFieldMap(u, rho, E, B);
  auto flux = abstractFlux->evaluateAt(exactMap);
  
  auto n = TFunction<double>::normal();
  FunctionPtr flux_dot_n;
  if (_spaceDim == 1) //flux is a scalar, then
  {
    auto n_x = n->spatialComponent(1);
    flux_dot_n = flux * n_x;
  }
  else
  {
    flux_dot_n = dot(_spaceDim,flux,n);
  }
  auto exactFlux = flux_dot_n;
  if (_spaceTime)
  {
    FunctionPtr n_t = TFunction<double>::normalSpaceTime()->t();
    auto timeTermExact = VarFunction<double>::abstractFunction(timeTerm)->evaluateAt(exactMap);
    exactFlux = exactFlux + timeTermExact * n_t;
  }
  if (includeParity)
  {
    exactFlux = exactFlux * Function::sideParity();
  }
  return exactFlux;
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionMap(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeFluxParity)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "exactSolutionMap not yet implemented!");
//  using namespace std;
//  vector<FunctionPtr> q(_spaceDim);
//  vector<vector<FunctionPtr>> D(_spaceDim,vector<FunctionPtr>(_spaceDim));
//  vector<FunctionPtr> u(_spaceDim);
//  FunctionPtr qWeight = (-Cp()/Pr())*_muFunc;
//  FunctionPtr E = rho * Cv() * T; // we add u*u/2 to this below
//  if (_spaceDim == 1)
//  {
//    D[0][0] = _muFunc * velocity->dx();
//    q[0] = qWeight * T->dx();
//    u[0] = velocity;
//    E = E + 0.5 * rho * u[0]*u[0];
//  }
//  else
//  {
//    for (int d1=0; d1<_spaceDim; d1++)
//    {
//      q[d1] = qWeight * T->di(d1+1);
//      u[d1] = velocity->spatialComponent(d1+1);
//      for (int d2=0; d2<_spaceDim; d2++)
//      {
//        D[d1][d2] = _muFunc * u[d1]->di(d2+1);
//      }
//      E = E + 0.5 * rho * u[d1] * u[d1];
//    }
//  }
//  vector<FunctionPtr> tm = exactSolution_tm(velocity, rho, T, includeFluxParity);
//  FunctionPtr         te = exactSolution_te(velocity, rho, T, includeFluxParity);
//  FunctionPtr         tc = exactSolution_tc(velocity, rho, T, includeFluxParity);
//
//  map<int, FunctionPtr> solnMap;
//  solnMap[this->E()->ID()]     = E;
//  solnMap[this->T_hat()->ID()] = T;
//  solnMap[this->rho()->ID()]   = rho;
//  solnMap[this->tc()->ID()]    = tc;
//  solnMap[this->te()->ID()]    = te;
//  for (int d1=0; d1<_spaceDim; d1++)
//  {
//    solnMap[this->m(d1+1)->ID()]     = u[d1] * rho;
//    solnMap[this->u_hat(d1+1)->ID()] = u[d1];
//
//    solnMap[this->q(d1+1)->ID()]     = q[d1];
//
//    solnMap[this->tm(d1+1)->ID()]    = tm[d1];
//
//    for (int d2=0; d2<_spaceDim; d2++)
//    {
//      solnMap[this->D(d1+1,d2+1)->ID()] = D[d1][d2];
//    }
//  }
//  return solnMap;
}

double IdealMHDFormulation::gamma()
{
  return _gamma;
}

int IdealMHDFormulation::getSolveCode()
{
  return _solveCode;
}

FunctionPtr IdealMHDFormulation::getTimeStep()
{
  return _dt;
}

double IdealMHDFormulation::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double IdealMHDFormulation::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

FunctionPtr IdealMHDFormulation::getMomentumFluxComponent(FunctionPtr momentumFlux, int i)
{
  // get the column (row in 1D) of the momentum tensor
  auto flux = (_spaceDim > 1) ? column(_spaceDim,momentumFlux,i+1) : momentumFlux->spatialComponent(i+1);
  return flux;
}

VarPtr IdealMHDFormulation::m(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_m[i-1]);
}

/*
double IdealMHDFormulation::Pr()
{
  return _Pr;
}

VarPtr IdealMHDFormulation::q(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_q[i-1]);
}

double IdealMHDFormulation::R()
{
  return Cp()-Cv();
}
*/
VarPtr IdealMHDFormulation::rho()
{
  return _vf->trialVar(S_rho);
}

RHSPtr IdealMHDFormulation::rhs()
{
  return _rhs;
}

void IdealMHDFormulation::setForcing(FunctionPtr f_continuity, std::vector<FunctionPtr> f_momentum, FunctionPtr f_energy,
                                     std::vector<FunctionPtr> f_magnetic)
{
  const int trueSpaceDim = 3;
  TEUCHOS_TEST_FOR_EXCEPTION(f_momentum.size() != trueSpaceDim, std::invalid_argument, "f_momentum should have size equal to 3, the true spatial dimension");
  TEUCHOS_TEST_FOR_EXCEPTION(f_magnetic.size() != trueSpaceDim, std::invalid_argument, "f_magnetic should have size equal to 3, the true spatial dimension");
  _fc->setValue(f_continuity);
  _fe->setValue(f_energy);
  for (int d=0; d<trueSpaceDim; d++)
  {
    _fm[d]->setValue(f_momentum[d]);
    _fB[d]->setValue(f_magnetic[d]);
  }
}

// ! set current time step used for transient solve
void IdealMHDFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double IdealMHDFormulation::solveAndAccumulate()
{
  _solveCode = _solnIncrement->solve(_solver);
  
  set<int> nonlinearVariables = {{rho()->ID(), E()->ID()}};
  set<int> linearVariables = {{tc()->ID(), te()->ID()}};
  
  const int trueSpaceDim = 3;
  for (int d1=0; d1<trueSpaceDim; d1++)
  {
    nonlinearVariables.insert(m(d1+1)->ID());
    linearVariables.insert(tm(d1+1)->ID());
  }
  
  double alpha = 1.0;
  ParameterFunctionPtr alphaParameter = ParameterFunction::parameterFunction(alpha);
  
  FunctionPtr rhoPrevious  = Function::solution(rho(),_backgroundFlow);
  FunctionPtr rhoIncrement = Function::solution(rho(),_solnIncrement);
  FunctionPtr rhoUpdated   = rhoPrevious + FunctionPtr(alphaParameter) * rhoIncrement;
  
  FunctionPtr EPrevious    = Function::solution(E(),_backgroundFlow);
  FunctionPtr EIncrement   = Function::solution(E(),_solnIncrement);
  FunctionPtr EUpdated     = EPrevious + FunctionPtr(alphaParameter) * EIncrement;
  
  vector<FunctionPtr> mPrevious(_spaceDim), mIncrement(_spaceDim), mUpdated(_spaceDim);
  FunctionPtr mDotmPrevious = Function::zero();
  FunctionPtr mDotmUpdated = Function::zero();
  for (int d=0; d<_spaceDim; d++)
  {
    mPrevious[d] = Function::solution(m(d+1),_backgroundFlow);
    mIncrement[d] = Function::solution(m(d+1),_solnIncrement);
    mUpdated[d] = mPrevious[d] + FunctionPtr(alphaParameter) * mIncrement[d];
    mDotmPrevious = mDotmPrevious + mPrevious[d] * mPrevious[d];
    mDotmUpdated  = mDotmUpdated  + mUpdated [d] * mUpdated [d];
  }
  
  double Cv = this->Cv();
  FunctionPtr TUpdated  = (1.0/Cv) / rhoUpdated  * (EUpdated  - 0.5 * mDotmUpdated  / rhoUpdated);
  FunctionPtr TPrevious = (1.0/Cv) / rhoPrevious * (EPrevious - 0.5 * mDotmPrevious / rhoPrevious);
  
  // we may need to do something else to ensure positive changes in entropy;
  // if we just add ds to the list of positive functions, we stall on the first Newton step...
  vector<FunctionPtr> positiveFunctions = {rhoUpdated, TUpdated};
  double minDistanceFromZero = 0.001; // "positive" values should not get *too* small...
  int posEnrich = 5;
  
  // lambda for positivity checking
  auto isPositive = [&] () -> bool
  {
    for (auto f : positiveFunctions)
    {
      FunctionPtr f_smaller = f - minDistanceFromZero;
      bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
      if (!isPositive) return false;
    }
    return true;
  };
  
  bool useLineSearch = true;
  
  if (useLineSearch)
  {
    double lineSearchFactor = .5;
    int iter = 0; int maxIter = 20;
    while (!isPositive() && iter < maxIter)
    {
      alpha = alpha*lineSearchFactor;
      alphaParameter->setValue(alpha);
      iter++;
    }
  }
  
  _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, nonlinearVariables, linearVariables);
  _nonlinearIterationCount++;
  
  return alpha;
}

// ! Returns the solution (at current time)
SolutionPtr IdealMHDFormulation::solution()
{
  return _backgroundFlow;
}

SolutionPtr IdealMHDFormulation::solutionIncrement()
{
  return _solnIncrement;
}

// ! Returns the solution (at previous time)
SolutionPtr IdealMHDFormulation::solutionPreviousTimeStep()
{
  return _solnPrevTime;
}

BFPtr IdealMHDFormulation::steadyBF()
{
  return _steadyBF;
}

VarPtr IdealMHDFormulation::tc()
{
  return _vf->trialVar(S_tc);
}

VarPtr IdealMHDFormulation::te()
{
  return _vf->trialVar(S_te);
}

VarPtr IdealMHDFormulation::tm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->trialVar(S_tm[i-1]);
}

VarPtr IdealMHDFormulation::vc()
{
  return _vf->testVar(S_vc);
}

VarPtr IdealMHDFormulation::vm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vm[i-1]);
}

VarPtr IdealMHDFormulation::ve()
{
  return _vf->testVar(S_ve);
}
