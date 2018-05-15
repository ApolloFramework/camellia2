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
const string IdealMHDFormulation::S_tGauss = "tGauss";

const string IdealMHDFormulation::S_vc = "vc";
const string IdealMHDFormulation::S_vm1  = "vm1";
const string IdealMHDFormulation::S_vm2  = "vm2";
const string IdealMHDFormulation::S_vm3  = "vm3";
const string IdealMHDFormulation::S_ve   = "ve";
const string IdealMHDFormulation::S_vB1  = "vB1";
const string IdealMHDFormulation::S_vB2  = "vB2";
const string IdealMHDFormulation::S_vB3  = "vB3";
const string IdealMHDFormulation::S_vGauss = "vGauss";

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

IdealMHDFormulation::IdealMHDFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _spaceDim = spaceDim;
  const int trueSpaceDim = 3; // as opposed to _spaceDim, which indicates the number of dimensions that can have non-zero derivatives (also the dimension of the mesh)

  _fc = ParameterFunction::parameterFunction(0.0);
  _fe = ParameterFunction::parameterFunction(0.0);
  _fm = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim);
  _fB = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim);
  _fc->setName("fc");
  _fe->setName("fe");
  for (int d=0; d<trueSpaceDim; d++)
  {
    _fm[d] = ParameterFunction::parameterFunction(0.0);
    _fB[d] = ParameterFunction::parameterFunction(0.0);
    ostringstream name_fm;
    name_fm << "fm" << d+1;
    _fm[d]->setName(name_fm.str());
    if ((d > 0) || (spaceDim > 1))
    {
      ostringstream name_fB;
      name_fB << "fB" << d+1;
      _fB[d]->setName(name_fB.str());
      cout << "set name of fB[" << d << "] to " << name_fB.str() << endl;
    }
  }
  _gamma = parameters.get<double>("gamma",2.0); // 2.0 is what is used for Brio-Wu
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
  
  // field variables
  VarPtr rhoVar;
  vector<VarPtr> mVar(trueSpaceDim);
  VarPtr EVar;
  vector<VarPtr> BVar(trueSpaceDim); // In 1D, first component will be null
  
  // trace variables
  VarPtr tc;
  vector<VarPtr> tm(trueSpaceDim);
  VarPtr te;
  vector<VarPtr> tB(trueSpaceDim); // first component null in 1D
  VarPtr tGauss; // only defined for spaceDim > 1
  
  // test variables
  VarPtr vc;
  vector<VarPtr> vm(trueSpaceDim);
  VarPtr ve;
  vector<VarPtr> vB(trueSpaceDim);
  VarPtr vGauss; // only defined for spaceDim > 1
  
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
  
  FunctionPtr B; // 3-component vector Function.  First component is concrete in 1D; others are abstract.
  vector<FunctionPtr> B_comps(trueSpaceDim);
  if (_spaceDim == 1)
  {
    // then the vars that remain as unknowns are By, Bz
    _Bx = ParameterFunction::parameterFunction(0.75); // (This is the value we use for Brio-Wu; can be over-ridden via setBx())
    _Bx->setName("Bx"); // used in displayString
    B_comps[0] = _Bx;
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
  
  // Gauss' Law, test and trial:  (Only need this for _spaceDim > 1)
  if (_spaceDim > 1)
  {
    vGauss = _vf->testVar(S_vGauss, HGRAD);
    tGauss = _vf->fluxVar(S_tGauss);
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
    
    _massFlux     = m;
    _momentumFlux = outerProduct(trueSpaceDim, u, m) + p_star * I - outerProduct(trueSpaceDim, B, B);
    _energyFlux   = (E + p_star) * u - B * dot(trueSpaceDim,B,u);
    _magneticFlux = outerProduct(trueSpaceDim, u, B) - outerProduct(trueSpaceDim, B, u);
    _gaussFlux    = B; // only used for spaceDim > 1
    
    if (_spaceDim == 1)
    {
      // take the x component of each flux term: this is the 1D flux (we drop y and z derivative terms)
      int x_comp = 1;
      _massFlux     = _massFlux->spatialComponent(x_comp);
      _momentumFlux = _momentumFlux->spatialComponent(x_comp);
      _energyFlux   = _energyFlux->spatialComponent(x_comp);
      _magneticFlux = _magneticFlux->spatialComponent(x_comp); // the first component of this is zero; we don't test against this in 1D
    }
    else if (_spaceDim == 2)
    {
      // take x,y components of each flux: this is 2D flux (we drop z derivative terms)
      int x_comp = 1, y_comp = 2;
      _massFlux     = Function::vectorize(_massFlux->spatialComponent    (x_comp),  _massFlux->spatialComponent    (y_comp));
      _momentumFlux = Function::vectorize(_momentumFlux->spatialComponent(x_comp),  _momentumFlux->spatialComponent(y_comp));
      _energyFlux   = Function::vectorize(_energyFlux->spatialComponent  (x_comp),  _energyFlux->spatialComponent  (y_comp));
      _magneticFlux = Function::vectorize(_magneticFlux->spatialComponent(x_comp),  _magneticFlux->spatialComponent(y_comp));
    }
    
    auto mDotm = dot(trueSpaceDim, m, m);
    auto Cv = this->Cv();
    auto gamma = this->gamma();
    _abstractTemperature = (1.0/Cv) / rho  * (E  - 0.5 * mDotm  / rho);
    _abstractPressure = (gamma - 1) * (E - 0.5 * mDotm / rho);
    _abstractVelocity = m / rho;
    _abstractMomentum = m;
    _abstractEnergy = E;
    _abstractDensity = rho;
    _abstractMagnetism = B;
    
    cout << "massFlux: " << _massFlux->displayString() << endl;
    cout << "energyFlux: " << _energyFlux->displayString() << endl;
    cout << "momentumFlux: " << _momentumFlux->displayString() << endl;
    cout << "magneticFlux: " << _magneticFlux->displayString() << endl;
  }
  
  _bf = Teuchos::rcp( new BF(_vf) );
  _steadyBF = Teuchos::rcp( new BF(_vf) );
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
  
  _fluxEquations[vc->ID()] = {vc,_massFlux,tc,rhoVar,_fc};
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto momentumColumn = (_spaceDim > 1) ? column(_spaceDim,_momentumFlux,d+1) : _momentumFlux->spatialComponent(d+1);
    {
      // A small tweak for 1D: cancel out the (B_x, B_x) component in the m_1 equation -- this is a constant.
      // It's likely just fine without this tweak: the placement of fluxPrevious on the RHS should take care of it.
      // Still, it's a slightly unusual thing to have a constant in the flux definition, and the fact that we are taking
      // the divergence means that mathematically getting rid of the constant is equivalent.
      if ((_spaceDim == 1) && (d==0))
      {
        momentumColumn = momentumColumn + B->x() * B->x();
      }
    }
    _fluxEquations[vm[d]->ID()] = {vm[d],momentumColumn,tm[d],mVar[d],_fm[d]};
    
    cout << "for variable " << mVar[d]->name() << ", flux is " << momentumColumn->displayString() << endl;
    if ((d > 0) || (_spaceDim != 1))
    {
      auto magneticColumn = (_spaceDim > 1) ? column(_spaceDim,_magneticFlux,d+1) : _magneticFlux->spatialComponent(d+1);
      _fluxEquations[vB[d]->ID()] = {vB[d],magneticColumn,tB[d],BVar[d],_fB[d]};
    }
  }
  _fluxEquations[ve->ID()] = {ve,_energyFlux,te,EVar,_fe};
  if (_spaceDim > 1)
  {
    // enforce Gauss' Law
    VarPtr timeTerm = Teuchos::null;
    _fluxEquations[vGauss->ID()] = {vGauss,_gaussFlux,tGauss,timeTerm,Function::zero()};
  }
  
  std::string backgroundFlowIdentifierExponent = "";
  std::string previousTimeIdentifierExponent = "";
  if (_timeStepping)
  {
    backgroundFlowIdentifierExponent = "k";
    previousTimeIdentifierExponent = "k-1";
  }
  // to avoid needing a bunch of casts below, do a cast once here:
  FunctionPtr dt = (FunctionPtr)_dt;
  
  bool weightFluxesByParity = true; // TODO: decide whether true or false is more appropriate (so far, we haven't used the DPG fluxes that are stored here...)
  for (auto varEntry : trialVars)
  {
    VarPtr var = varEntry.second;
    _backgroundFlowMap  [var->ID()] = Function::solution(var, _backgroundFlow, weightFluxesByParity, backgroundFlowIdentifierExponent);
    _solnPreviousTimeMap[var->ID()] = Function::solution(var, _solnPrevTime,   weightFluxesByParity, previousTimeIdentifierExponent);
    _solnIncrementMap   [var->ID()] = Function::solution(var, _solnIncrement,  weightFluxesByParity, "");
  }
  
  for (auto eqnEntry : _fluxEquations)
  {
    auto eqn = eqnEntry.second;
    auto testVar  = eqn.testVar;
    auto timeTerm = eqn.timeTerm;
    auto flux     = eqn.flux;
    auto traceVar = eqn.traceVar;
    auto f_rhs    = eqn.f_rhs;
    {
      // DEBUGGING
      cout << "testVar:  " << testVar->name() << endl;
      if (timeTerm != Teuchos::null) cout << "timeTerm: " << timeTerm->name() << endl;
      cout << "flux:     " << flux->displayString() << endl;
      cout << "traceVar: " << traceVar->name() << endl;
      cout << "f_rhs:    " << f_rhs->displayString() << endl;
    }
    if (timeTerm != Teuchos::null) // time term is null for Gauss' Law
    {
      auto timeTerm_prev      = _backgroundFlowMap.find(timeTerm->ID())->second;
      auto timeTerm_prev_time = _solnPreviousTimeMap.find(timeTerm->ID())->second;
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
    }
    auto fluxPrevious = flux->evaluateAt(_backgroundFlowMap);
    auto fluxJacobian = flux->jacobian(_backgroundFlowMap);
    Camellia::EOperator gradOp = (_spaceDim > 1) ? OP_GRAD : OP_DX;
    
    _bf->      addTerm(-fluxJacobian, testVar->applyOp(gradOp)); // negative from integration by parts
    _steadyBF->addTerm(-fluxJacobian, testVar->applyOp(gradOp));
    _rhs     ->addTerm(fluxPrevious * testVar->applyOp(gradOp));
    
    _bf->      addTerm(traceVar, testVar);
    _steadyBF->addTerm(traceVar, testVar);
    _rhs     ->addTerm(f_rhs * testVar);
  }
  
  // DEBUGGING
  cout << "bf: " << _bf->displayString() << endl;
  
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

FunctionPtr IdealMHDFormulation::abstractDensity() const
{
  return _abstractDensity;
}

FunctionPtr IdealMHDFormulation::abstractEnergy() const
{
  return _abstractEnergy;
}

FunctionPtr IdealMHDFormulation::abstractMagnetism() const
{
  return _abstractMagnetism;
}

FunctionPtr IdealMHDFormulation::abstractMomentum() const
{
  return _abstractMomentum;
}

FunctionPtr IdealMHDFormulation::abstractPressure() const
{
  return _abstractPressure;
}

FunctionPtr IdealMHDFormulation::abstractTemperature() const
{
  return _abstractTemperature;
}

FunctionPtr IdealMHDFormulation::abstractVelocity() const
{
  return _abstractVelocity;
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

void IdealMHDFormulation::addMagneticFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tB_exact = exactSolution_tB(rho, u, E, B, includeParity);
  int trueSpaceDim = 3;
  int dStart = (_spaceDim > 1) ? 0 : 1; // B_x is not a solution variable in 1D
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    VarPtr tB_i = this->tB(d+1);
    _solnIncrement->bc()->addDirichlet(tB_i, region, tB_exact[d]);
    cout << "adding boundary condition tB" << d+1 << " = " << tB_exact[d]->displayString() << endl;
  }
}

void IdealMHDFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  VarPtr tc = this->tc();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tc_exact = this->exactSolution_tc(rho, u, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
  cout << "adding boundary condition tc = " << tc_exact->displayString() << endl;
}

void IdealMHDFormulation::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tm_exact = exactSolution_tm(rho, u, E, B, includeParity);
  int trueSpaceDim = 3;
  for (int d=0; d<trueSpaceDim; d++)
  {
    VarPtr tm_i = this->tm(d+1);
    _solnIncrement->bc()->addDirichlet(tm_i, region, tm_exact[d]);
    cout << "adding boundary condition tm" << d+1 << " = " << tm_exact[d]->displayString() << endl;
  }
}

void IdealMHDFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  VarPtr te = this->te();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto te_exact = exactSolution_te(rho, u, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
  cout << "adding boundary condition te = " << te_exact->displayString() << endl;
}

VarPtr IdealMHDFormulation::B(int i)
{
  return _vf->trialVar(S_B[i-1]);
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

// ! For an exact solution (rho, u, E, B), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_fB(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _magneticFlux;
  
  vector<FunctionPtr> f(3);
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto fluxes = abstractFlux->evaluateAt(exactMap);
  
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int i=dStart; i<3; i++)
  {
    // get the column (row in 1D) of the magnetism tensor
    auto flux = (_spaceDim > 1) ? column(_spaceDim,fluxes,i+1) : fluxes->spatialComponent(i+1);
    
    FunctionPtr timeTerm = B->spatialComponent(i+1); // the thing that gets differentiated in time
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

// ! For an exact solution (rho, u, E, B), returns the corresponding forcing in the continuity equation
FunctionPtr IdealMHDFormulation::exactSolution_fc(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _massFlux;
  FunctionPtr timeTerm = rho; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);

  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
FunctionPtr IdealMHDFormulation::exactSolution_fe(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _energyFlux;
  FunctionPtr timeTerm = E; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);
  
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_fm(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _momentumFlux;
  
  vector<FunctionPtr> f(3);

  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
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

std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_tB(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  const int trueSpaceDim = 3;
  std::vector<FunctionPtr> tB(trueSpaceDim);
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    auto testVar = this->vB(d+1);
    tB[d] = exactSolutionFlux(testVar, rho, u, E, B, includeParity);
  }
  return tB;
}

FunctionPtr IdealMHDFormulation::exactSolution_tc(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->vc();
  return exactSolutionFlux(testVar, rho, u, E, B, includeParity);
}

FunctionPtr IdealMHDFormulation::exactSolution_te(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->ve();
  return exactSolutionFlux(testVar, rho, u, E, B, includeParity);
}

std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_tm(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  const int trueSpaceDim = 3;
  std::vector<FunctionPtr> tm(trueSpaceDim);
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto testVar = this->vm(d+1);
    tm[d] = exactSolutionFlux(testVar, rho, u, E, B, includeParity);
  }
  return tm;
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionFieldMap(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(u->rank() != 1, std::invalid_argument, "u must be a vector-valued function");
  TEUCHOS_TEST_FOR_EXCEPTION(B->rank() != 1, std::invalid_argument, "B must be a vector-valued function");
  
  return exactSolutionFieldMapFromConservationVariables(rho, u*rho, E, B);
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionFieldMapFromConservationVariables(FunctionPtr rho, FunctionPtr m, FunctionPtr E, FunctionPtr B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(m->rank() != 1, std::invalid_argument, "m must be a vector-valued function");
  TEUCHOS_TEST_FOR_EXCEPTION(B->rank() != 1, std::invalid_argument, "B must be a vector-valued function");
  
  map<int, FunctionPtr> exactMap;
  const int trueSpaceDim = 3;
  exactMap[this->rho()->ID()] = rho;
  exactMap[this->E()->ID()] = E;
  for (int d=1; d<=trueSpaceDim; d++)
  {
    exactMap[this->m(d)->ID()] = m->spatialComponent(d);
    if ((d > 1) || (_spaceDim > 1))
    {
      exactMap[this->B(d)->ID()] = B->spatialComponent(d);
    }
  }
  return exactMap;
}

FunctionPtr IdealMHDFormulation::exactSolutionFlux(VarPtr testVar, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B,
                                                   bool includeParity)
{
  ScalarFluxEquation equation = _fluxEquations.find(testVar->ID())->second;
  auto abstractFlux = equation.flux;
  auto timeTerm = equation.timeTerm;
  
  cout << "u: " << u->displayString() << endl;
  map<int,FunctionPtr> exactMap = exactSolutionFieldMap(rho, u, E, B);
  {
    // DEBUGGING
    cout << "exactMap:\n";
    for (auto entry : exactMap)
    {
      VarPtr var = _vf->trial(entry.first);
      cout << var->name() << " -> " << entry.second->displayString() << endl;
    }
    cout << "abstract flux for test variable " << testVar->name() << ": " << abstractFlux->displayString() << endl;
  }
  
  auto flux = abstractFlux->evaluateAt(exactMap);
  
  cout << "Concrete flux: " << flux->displayString() << endl;
  
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

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionMap(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeFluxParity)
{
  map<int, FunctionPtr> exactMap;
  
  auto varE   = this->E();
  auto varRho = this->rho();
  
  exactMap[varRho->ID()] = rho;
  exactMap[varE->ID()]   = E;
  exactMap[this->tc()->ID()] = this->exactSolution_tc(rho, u, E, B, includeFluxParity);
  exactMap[this->te()->ID()] = this->exactSolution_te(rho, u, E, B, includeFluxParity);
  
  int trueSpaceDim = 3;
  auto tmExact = this->exactSolution_tm(rho, u, E, B, includeFluxParity);
  for (int d=0; d<trueSpaceDim; d++)
  {
    exactMap[this->m(d+1)->ID()]  = u->spatialComponent(d+1) * rho;
    exactMap[this->tm(d+1)->ID()] = tmExact[d];
  }
  
  int dStart = (_spaceDim > 1) ? 0 : 1;
  auto tBExact = this->exactSolution_tB(rho, u, E, B, includeFluxParity);
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    exactMap[this->B(d+1)->ID()]  = B->spatialComponent(d+1);
    exactMap[this->tB(d+1)->ID()] = tBExact[d];
  }
  
  return exactMap;
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
 
double IdealMHDFormulation::R()
{
  return Cp()-Cv();
}

VarPtr IdealMHDFormulation::rho()
{
  return _vf->trialVar(S_rho);
}

RHSPtr IdealMHDFormulation::rhs()
{
  return _rhs;
}

void IdealMHDFormulation::setBx(FunctionPtr Bx)
{
  if (_spaceDim != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setBx() is only supported for 1D");
  }
  _Bx->setValue(Bx);
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

void IdealMHDFormulation::setInitialState(const std::map<int, FunctionPtr> &initialState)
{
  const int solutionOrdinal = 0;
  _backgroundFlow->projectOntoMesh(initialState, solutionOrdinal);
  if (!_spaceTime)
  {
    _solnPrevTime->projectOntoMesh(initialState, solutionOrdinal);
  }
}

// ! set current time step used for transient solve
void IdealMHDFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double IdealMHDFormulation::solveAndAccumulate()
{
//  cout << "_solnIncrement->bf(): " << _solnIncrement->bf()->displayString() << endl;
  
  _solveCode = _solnIncrement->solve(_solver);
  
  set<int> nonlinearVariables = {{rho()->ID(), E()->ID()}};
  set<int> linearVariables = {{tc()->ID(), te()->ID()}};
  
  const int trueSpaceDim = 3;
  for (int d1=0; d1<trueSpaceDim; d1++)
  {
    nonlinearVariables.insert(m(d1+1)->ID());
    linearVariables.insert(tm(d1+1)->ID());
    if ((d1>0) || (_spaceDim > 1))
    {
      nonlinearVariables.insert(B(d1+1)->ID());
      linearVariables.insert(tB(d1+1)->ID());
    }
  }
  
  double alpha = 1.0;
  ParameterFunctionPtr alphaParameter = ParameterFunction::parameterFunction(alpha);
  alphaParameter->setName("alpha");
  
  map<int,FunctionPtr> updatedFieldMap;
  for (auto entry : _solnPreviousTimeMap)
  {
    auto uID = entry.first;
    auto uPrev = entry.second;
    FunctionPtr uIncr = _solnIncrementMap.find(uID)->second;
    FunctionPtr uUpdated = uPrev + FunctionPtr(alphaParameter) * uIncr;
    updatedFieldMap[uID] = uUpdated;
  }

  auto rhoUpdated = updatedFieldMap.find(this->rho()->ID())->second;
  auto TUpdated   = _abstractTemperature->evaluateAt(updatedFieldMap);
  
//  cout << "TUpdated: " << TUpdated->displayString() << endl;
//  cout << "rhoUpdated: " << rhoUpdated->displayString() << endl;
  
  // we may need to do something else to ensure positive changes in entropy;
  // if we just add ds to the list of positive functions, we stall on the first Newton step...
  vector<FunctionPtr> positiveFunctions = {rhoUpdated, TUpdated};
//  {
//    // DEBUGGING:
//    cout << "WARNING: temporarily (DEBUGGING) turning off positivity enforcement.\n";
//    positiveFunctions = {};
//  }
  double minDistanceFromZero = 0.000000001; // "positive" values should not get *too* small...
  int posEnrich = 5;
  
  // lambda for positivity checking
  auto isPositive = [&] () -> bool
  {
    for (auto f : positiveFunctions)
    {
      FunctionPtr f_smaller = f - minDistanceFromZero;
      bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
      if (!isPositive)
      {
//        cout << "function " << f->displayString() << " is not positive for alpha = " << alphaParameter->getValue()->displayString() << endl;
        return false;
      }
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

// ! Returns a map containing the current time step's solution data for each trial variable
const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionFieldMap() const
{
  return _backgroundFlowMap;
}

SolutionPtr IdealMHDFormulation::solutionIncrement()
{
  return _solnIncrement;
}

const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionIncrementFieldMap() const
{
  return _solnIncrementMap;
}


// ! Returns the solution (at previous time)
SolutionPtr IdealMHDFormulation::solutionPreviousTimeStep()
{
  return _solnPrevTime;
}

const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionPreviousTimeStepFieldMap() const
{
  return _solnPreviousTimeMap;
}

BFPtr IdealMHDFormulation::steadyBF()
{
  return _steadyBF;
}

VarPtr IdealMHDFormulation::tB(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->trialVar(S_tB[i-1]);
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

VarPtr IdealMHDFormulation::vB(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vB[i-1]);
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
