//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  OldroydBFormulation.cpp
//  Camellia
//
//  Created by B. Keith 10/15.
//
//

#include "OldroydBFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"

using namespace Camellia;

OldroydBFormulation OldroydBFormulation::steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                         MeshTopologyPtr meshTopo, int polyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useConservationFormulation",false);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("spatialPolyOrder", polyOrder);
  parameters.set("delta_k", delta_k);

  return OldroydBFormulation(meshTopo, parameters);
}

// OldroydBFormulation OldroydBFormulation::spaceTimeFormulation(int spaceDim, double Re, bool useConformingTraces,
//                                                                             MeshTopologyPtr meshTopo, int spatialPolyOrder, int temporalPolyOrder, int delta_k)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",1.0 / Re);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useConservationFormulation",false);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", true);

//   parameters.set("includeVelocityTracesInFluxTerm",false);
//   parameters.set("t0",0.0);

//   parameters.set("spatialPolyOrder", spatialPolyOrder);
//   parameters.set("temporalPolyOrder", temporalPolyOrder);
//   parameters.set("delta_k", delta_k);

//   //  {
//   //    // DEBUGGING:
//   //    cout << "OldroydBFormulation: adding 1 to the tn1hat poly order.\n";
//   //    string var_adjustString = S_TN1_HAT + "-polyOrderAdjustment";
//   //    parameters.set(var_adjustString, 1); // add 1 to the poly order of variable
//   //  }

//   return OldroydBFormulation(meshTopo, parameters);
// }

// OldroydBFormulation OldroydBFormulation::steadyConservationFormulation(int spaceDim, double Re, bool useConformingTraces,
//                                                                          MeshTopologyPtr meshTopo, int polyOrder, int delta_k)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",1.0 / Re);
//   parameters.set("useConservationFormulation",true);
//   parameters.set("includeVelocityTracesInFluxTerm",true);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", false);
//   parameters.set("spatialPolyOrder", polyOrder);
//   parameters.set("delta_k", delta_k);

//   return OldroydBFormulation(meshTopo, parameters);
// }

// OldroydBFormulation OldroydBFormulation::spaceTimeConservationFormulation(int spaceDim, double Re, bool useConformingTraces,
//                                                                             MeshTopologyPtr meshTopo, int spatialPolyOrder, int temporalPolyOrder, int delta_k)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",1.0 / Re);
//   parameters.set("useConservationFormulation",true);
//   parameters.set("includeVelocityTracesInFluxTerm",true);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", true);

//   parameters.set("includeVelocityTracesInFluxTerm",false);
//   parameters.set("t0",0.0);

//   parameters.set("spatialPolyOrder", spatialPolyOrder);
//   parameters.set("temporalPolyOrder", temporalPolyOrder);
//   parameters.set("delta_k", delta_k);

//   //  {
//   //    // DEBUGGING:
//   //    cout << "OldroydBFormulation: adding 1 to the tn1hat poly order.\n";
//   //    string var_adjustString = S_TN1_HAT + "-polyOrderAdjustment";
//   //    parameters.set(var_adjustString, 1); // add 1 to the poly order of variable
//   //  }

//   return OldroydBFormulation(meshTopo, parameters);
// }


// OldroydBFormulation OldroydBFormulation::timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
//                                                                                MeshTopologyPtr meshTopo, int polyOrder, int delta_k,
//                                                                                double dt, TimeStepType timeStepType)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",1.0 / Re);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", true);
//   parameters.set("useSpaceTime", false);
//   parameters.set("useConservationFormulation",false);
//   parameters.set("dt", dt);
//   parameters.set("timeStepType", timeStepType);

//   parameters.set("spatialPolyOrder", polyOrder);
//   parameters.set("delta_k", delta_k);

//   return OldroydBFormulation(meshTopo, parameters);
// }

OldroydBFormulation::OldroydBFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _stokesForm = Teuchos::rcp( new StokesVGPFormulation(parameters) );
  _spaceDim = parameters.get<int>("spaceDim");
  _conservationFormulation = parameters.get<bool>("useConservationFormulation");
  _neglectFluxesOnRHS = true;

  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string filePrefix = parameters.get<string>("fileToLoadPrefix", "");

  vector<int> H1Order;
  if (_stokesForm->isSpaceTime())
  {
    H1Order = {spatialPolyOrder+1,temporalPolyOrder+1}; // not dead certain that temporalPolyOrder+1 is the best choice; it depends on whether the indicated poly order means L^2 as it does in space, or whether it means H^1...
  }
  else
  {
    H1Order = {spatialPolyOrder+1};
  }

  map<int,int> trialVariableAdjustments = _stokesForm->getTrialVariablePolyOrderAdjustments();
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, _stokesForm->bf(), H1Order, delta_k, trialVariableAdjustments) ) ;

  _backgroundFlow = TSolution<double>::solution(mesh);
  _solnIncrement = TSolution<double>::solution(mesh);

  // want to enrich the cubature to match max background flow degree for sigma and/or u field variables:
  int maxBackgroundFlowDegree = spatialPolyOrder;
  for (int comp_i=1; comp_i<=_spaceDim; comp_i++)
  {
    VarPtr u_i = this->u(comp_i);

    if (trialVariableAdjustments.find(u_i->ID()) != trialVariableAdjustments.end())
    {
      int adjustment = trialVariableAdjustments[u_i->ID()];
      maxBackgroundFlowDegree = max(maxBackgroundFlowDegree, spatialPolyOrder + adjustment);
    }

    for (int comp_j=1; comp_j<=_spaceDim; comp_j++)
    {
      VarPtr sigma_ij = this->sigma(comp_i, comp_j);
      if (trialVariableAdjustments.find(sigma_ij->ID()) != trialVariableAdjustments.end())
      {
        int adjustment = trialVariableAdjustments[sigma_ij->ID()];
        maxBackgroundFlowDegree = max(maxBackgroundFlowDegree, spatialPolyOrder + adjustment);
      }
    }
  }
  _solnIncrement->setCubatureEnrichmentDegree(maxBackgroundFlowDegree);

  mesh->registerSolution(_backgroundFlow); // will project background flow during refinements...
  mesh->registerSolution(_solnIncrement);

  // copy the Stokes BF thus far for modification for Navier-Stokes
  _navierStokesBF = Teuchos::rcp( new BF(*_stokesForm->bf()) );

  // to avoid circular references, all previous solution references in BF won't own the memory:
  TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );

  // convective terms:
  vector<FunctionPtr> sigma_prev, u_prev;

  double Re = 1.0 / _stokesForm->mu();

  TFunctionPtr<double> p_prev = TFunction<double>::solution(_stokesForm->p(), backgroundFlowWeakReference);
  if (!_conservationFormulation)
  {
    for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
    {
      VarPtr u_i = _stokesForm->u(comp_i);
      VarPtr v_i = _stokesForm->v(comp_i);

      for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
      {
        VarPtr u_j = _stokesForm->u(comp_j);
        VarPtr sigma_ij = _stokesForm->sigma(comp_i, comp_j);

        FunctionPtr sigma_prev_ij = TFunction<double>::solution(sigma_ij, backgroundFlowWeakReference);
        FunctionPtr u_prev_j = TFunction<double>::solution(u_j, backgroundFlowWeakReference);

        _navierStokesBF->addTerm( Re * sigma_prev_ij * u_j, v_i);
        _navierStokesBF->addTerm( Re * u_prev_j * sigma_ij, v_i);
      }
    }
  }
  else
  {
    if (_spaceDim == 2)
    {
      VarPtr u_1 = _stokesForm->u(1);
      VarPtr u_2 = _stokesForm->u(2);
      VarPtr v_1 = _stokesForm->v(1);
      VarPtr v_2 = _stokesForm->v(2);
      FunctionPtr u_prev_1 = TFunction<double>::solution(u_1, backgroundFlowWeakReference);
      FunctionPtr u_prev_2 = TFunction<double>::solution(u_2, backgroundFlowWeakReference);

      _navierStokesBF->addTerm(-u_prev_1*u_1, v_1->dx());
      _navierStokesBF->addTerm(-u_prev_1*u_1, v_1->dx());
      _navierStokesBF->addTerm(-u_prev_2*u_1, v_1->dy());
      _navierStokesBF->addTerm(-u_prev_1*u_2, v_1->dy());

      _navierStokesBF->addTerm(-u_prev_2*u_1, v_2->dx());
      _navierStokesBF->addTerm(-u_prev_1*u_2, v_2->dx());
      _navierStokesBF->addTerm(-u_prev_2*u_2, v_2->dy());
      _navierStokesBF->addTerm(-u_prev_2*u_2, v_2->dy());
    }
    else if (_spaceDim == 3)
    {
      VarPtr u_1 = _stokesForm->u(1);
      VarPtr u_2 = _stokesForm->u(2);
      VarPtr u_3 = _stokesForm->u(3);
      VarPtr v_1 = _stokesForm->v(1);
      VarPtr v_2 = _stokesForm->v(2);
      VarPtr v_3 = _stokesForm->v(3);
      FunctionPtr u_prev_1 = TFunction<double>::solution(u_1, backgroundFlowWeakReference);
      FunctionPtr u_prev_2 = TFunction<double>::solution(u_2, backgroundFlowWeakReference);
      FunctionPtr u_prev_3 = TFunction<double>::solution(u_3, backgroundFlowWeakReference);

      _navierStokesBF->addTerm(u_prev_1*u_1, v_1->dx());
      _navierStokesBF->addTerm(u_prev_1*u_1, v_1->dx());
      _navierStokesBF->addTerm(u_prev_2*u_1, v_1->dy());
      _navierStokesBF->addTerm(u_prev_1*u_2, v_1->dy());
      _navierStokesBF->addTerm(u_prev_3*u_1, v_1->dz());
      _navierStokesBF->addTerm(u_prev_1*u_3, v_1->dz());

      _navierStokesBF->addTerm(u_prev_1*u_2, v_2->dx());
      _navierStokesBF->addTerm(u_prev_2*u_1, v_2->dx());
      _navierStokesBF->addTerm(u_prev_2*u_2, v_2->dy());
      _navierStokesBF->addTerm(u_prev_2*u_2, v_2->dy());
      _navierStokesBF->addTerm(u_prev_3*u_2, v_2->dz());
      _navierStokesBF->addTerm(u_prev_2*u_3, v_2->dz());

      _navierStokesBF->addTerm(u_prev_1*u_3, v_3->dx());
      _navierStokesBF->addTerm(u_prev_3*u_1, v_3->dx());
      _navierStokesBF->addTerm(u_prev_2*u_3, v_3->dy());
      _navierStokesBF->addTerm(u_prev_3*u_2, v_3->dy());
      _navierStokesBF->addTerm(u_prev_3*u_3, v_3->dz());
      _navierStokesBF->addTerm(u_prev_3*u_3, v_3->dz());
    }
  }

  // cout << endl << _navierStokesBF->displayString() << endl;

  // set the inner product to the graph norm:
  // TODO: make this more general (pass norm name in via parameters)
  setIP( _navierStokesBF->graphNorm() );

  this->setForcingFunction(Teuchos::null); // will default to zero

  _bc = BC::bc();

  _solnIncrement->setBC(_bc);

  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );

  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );

  // Set up Functions for L^2 norm computations

  TFunctionPtr<double> p_incr = TFunction<double>::solution(_stokesForm->p(), _solnIncrement);
  p_prev = TFunction<double>::solution(_stokesForm->p(), _backgroundFlow);

  _L2IncrementFunction = p_incr * p_incr;
  _L2SolutionFunction = p_prev * p_prev;
  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    TFunctionPtr<double> u_i_incr = TFunction<double>::solution(_stokesForm->u(comp_i), _solnIncrement);
    TFunctionPtr<double> u_i_prev = TFunction<double>::solution(_stokesForm->u(comp_i), _backgroundFlow);

    _L2IncrementFunction = _L2IncrementFunction + u_i_incr * u_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + u_i_prev * u_i_prev;

    for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
    {
      TFunctionPtr<double> sigma_ij_incr = TFunction<double>::solution(_stokesForm->sigma(comp_i,comp_j), _solnIncrement);
      TFunctionPtr<double> sigma_ij_prev = TFunction<double>::solution(_stokesForm->sigma(comp_i,comp_j), _backgroundFlow);
      _L2IncrementFunction = _L2IncrementFunction + sigma_ij_incr * sigma_ij_incr;
      _L2SolutionFunction = _L2SolutionFunction + sigma_ij_prev * sigma_ij_prev;
    }
  }

  _solver = Solver::getDirectSolver();

  _nonlinearIterationCount = 0;

  if (_spaceDim==2)
  {
    // finally, set up a stream function solve for 2D
    _streamFormulation = Teuchos::rcp( new PoissonFormulation(_spaceDim,_useConformingTraces) );

    MeshPtr streamMesh;
    if (filePrefix == "")
    {
      MeshTopologyPtr streamMeshTopo = meshTopo->deepCopy();
      streamMesh = Teuchos::rcp( new Mesh(streamMeshTopo, _streamFormulation->bf(), H1Order, delta_k) ) ;
    }
    else
    {
      streamMesh = MeshFactory::loadFromHDF5(_streamFormulation->bf(), filePrefix+"_stream.mesh");
    }

    mesh->registerObserver(streamMesh); // refine streamMesh whenever mesh is refined

    LinearTermPtr u1_dy = Re * this->sigma(1,2);
    LinearTermPtr u2_dx = Re * this->sigma(2,1);

    TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_backgroundFlow, u2_dx - u1_dy) );
    RHSPtr streamRHS = RHS::rhs();
    VarPtr q_stream = _streamFormulation->q();
    streamRHS->addTerm( -vorticity * q_stream );
    bool dontWarnAboutOverriding = true;
    ((PreviousSolutionFunction<double>*) vorticity.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    /* Stream function phi is such that
     *    d/dx phi = -u2
     *    d/dy phi =  u1
     * Therefore, psi = grad phi = (-u2, u1), and psi * n = u1 n2 - u2 n1
     */

    TFunctionPtr<double> u1_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_backgroundFlow, this->u(1) ) );
    TFunctionPtr<double> u2_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_backgroundFlow, this->u(2) ) );
    ((PreviousSolutionFunction<double>*) u1_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    ((PreviousSolutionFunction<double>*) u2_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    TFunctionPtr<double> n = TFunction<double>::normal();

    BCPtr streamBC = BC::bc();
    VarPtr phi = _streamFormulation->phi();
    streamBC->addZeroMeanConstraint(phi);

    VarPtr psi_n = _streamFormulation->psi_n_hat();
    streamBC->addDirichlet(psi_n, SpatialFilter::allSpace(), u1_soln * n->y() - u2_soln * n->x());

    IPPtr streamIP = _streamFormulation->bf()->graphNorm();
    _streamSolution = TSolution<double>::solution(streamMesh,streamBC,streamRHS,streamIP);

    if (filePrefix != "")
    {
      _streamSolution->loadFromHDF5(filePrefix + "_stream.soln");
    }
  }
}

void OldroydBFormulation::addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u)
{
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getDimension();

  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (spaceDim==3) u3_hat = this->u_hat(3);

  if (_neglectFluxesOnRHS)
  {
    // this also governs how we accumulate in the fluxes and traces, and hence whether we should use zero BCs or the true BCs for solution increment
    _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
    _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u->y());
    if (spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, inflowRegion, u->z());
  }
  else
  {
    // we assume that _neglectFluxesOnRHS = true, in that we always use the full BCs, not their zero-imposing counterparts, when solving for solution increment
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_neglectFluxesOnRHS = true assumed various places");

    TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );

    TFunctionPtr<double> u1_hat_prev = TFunction<double>::solution(u1_hat,backgroundFlowWeakReference);
    TFunctionPtr<double> u2_hat_prev = TFunction<double>::solution(u2_hat,backgroundFlowWeakReference);
    TFunctionPtr<double> u3_hat_prev;
    if (spaceDim == 3) u3_hat_prev = TFunction<double>::solution(u3_hat,backgroundFlowWeakReference);

    _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u->x() - u1_hat_prev);
    _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u->y() - u2_hat_prev);
    if (spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, inflowRegion, u->z() - u3_hat_prev);
  }
}

void OldroydBFormulation::addOutflowCondition(SpatialFilterPtr outflowRegion)
{
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getDimension();

  // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solnIncrement->filter();

  Teuchos::RCP< PenaltyConstraints > pcRCP;
  PenaltyConstraints* pc;

  if (filter_incr.get() != NULL)
  {
    pc = dynamic_cast<PenaltyConstraints*>(filter_incr.get());
    if (pc == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't add PenaltyConstraints when a non-PenaltyConstraints LocalStiffnessMatrixFilter already in place");
    }
  }
  else
  {
    pcRCP = Teuchos::rcp( new PenaltyConstraints );
    pc = pcRCP.get();
  }
  TFunctionPtr<double> zero = TFunction<double>::zero();
  pc->addConstraint(_stokesForm->getTraction(1)==zero, outflowRegion);
  pc->addConstraint(_stokesForm->getTraction(2)==zero, outflowRegion);
  if (spaceDim==3) pc->addConstraint(_stokesForm->getTraction(3)==zero, outflowRegion);

  if (pcRCP != Teuchos::null)   // i.e., we're not just adding to a prior PenaltyConstraints object
  {
    _solnIncrement->setFilter(pcRCP);
  }
}

void OldroydBFormulation::addPointPressureCondition()
{
  VarPtr p = this->p();

  _solnIncrement->bc()->addSinglePointBC(p->ID(), 0.0, _solnIncrement->mesh());

  if (_solnIncrement->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    _solnIncrement->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void OldroydBFormulation::addWallCondition(SpatialFilterPtr wall)
{
  int spaceDim = _solnIncrement->mesh()->getTopology()->getDimension();
  vector<double> zero(spaceDim, 0.0);
  addInflowCondition(wall, TFunction<double>::constant(zero));
}

void OldroydBFormulation::addZeroMeanPressureCondition()
{
  VarPtr p = this->p();

  _solnIncrement->bc()->addZeroMeanConstraint(p);

  if (_solnIncrement->bc()->singlePointBC(p->ID()))
  {
    _solnIncrement->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr OldroydBFormulation::bf()
{
  return _navierStokesBF;
}

FunctionPtr OldroydBFormulation::convectiveTerm(int spaceDim, FunctionPtr u_exact)
{
  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 2) && (spaceDim != 3), std::invalid_argument, "spaceDim must be 2 or 3");

  TFunctionPtr<double> f;

  vector<FunctionPtr> convectiveTermVector(spaceDim, Function::zero());
  for (int i=1; i<=spaceDim; i++)
  {
    FunctionPtr ui_exact;
    switch (i) {
      case 1:
        ui_exact = u_exact->x();
        break;
      case 2:
        ui_exact = u_exact->y();
        break;
      case 3:
        ui_exact = u_exact->z();
        break;

      default:
        break;
    }
    for (int j=1; j<=spaceDim; j++)
    {
      FunctionPtr ui_dj_exact;
      switch (j) {
        case 1:
          ui_dj_exact = ui_exact->dx();
          break;
        case 2:
          ui_dj_exact = ui_exact->dy();
          break;
        case 3:
          ui_dj_exact = ui_exact->dz();
          break;

        default:
          break;
      }
      FunctionPtr uj_exact;
      switch (j) {
        case 1:
          uj_exact = u_exact->x();
          break;
        case 2:
          uj_exact = u_exact->y();
          break;
        case 3:
          uj_exact = u_exact->z();
          break;

        default:
          break;
      }

      convectiveTermVector[i-1] = convectiveTermVector[i-1] + uj_exact * ui_dj_exact;
    }
  }
  if (spaceDim == 2)
  {
    return Function::vectorize(convectiveTermVector[0],convectiveTermVector[1]);
  }
  else
  {
    return Function::vectorize(convectiveTermVector[0],convectiveTermVector[1],convectiveTermVector[2]);
  }
}

Teuchos::RCP<ExactSolution<double>> OldroydBFormulation::exactSolution(TFunctionPtr<double> u_exact, TFunctionPtr<double> p_exact)
{
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getDimension();

  // f1 and f2 are those for Stokes, but minus u \cdot \grad u
  TFunctionPtr<double> mu = TFunction<double>::constant(_stokesForm->mu());

  double Re = 1.0 / _stokesForm->mu();
  TFunctionPtr<double> f = OldroydBFormulation::forcingFunction(spaceDim, Re, u_exact, p_exact);

  VarPtr p = this->p();

  BCPtr bc = BC::bc();
  bc->addSinglePointBC(p->ID(), 0.0, _backgroundFlow->mesh());
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  for (int comp_i=1; comp_i<= spaceDim; comp_i++)
  {
    FunctionPtr ui_exact = u_exact->spatialComponent(comp_i);
    VarPtr ui_hat = this->u_hat(comp_i);
    bc->addDirichlet(ui_hat, boundary, ui_exact);
  }

  RHSPtr rhs = this->rhs(f,false);
  Teuchos::RCP<ExactSolution<double>> mySolution = Teuchos::rcp( new ExactSolution<double>(_navierStokesBF, bc, rhs) );

  mySolution->setSolutionFunction(p, p_exact);

  TFunctionPtr<double> sideParity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();

  for (int comp_i=1; comp_i <= spaceDim; comp_i++)
  {
    FunctionPtr ui_exact = u_exact->spatialComponent(comp_i);
    VarPtr ui = this->u(comp_i);
    mySolution->setSolutionFunction(ui, ui_exact);

    VarPtr ui_hat = this->u_hat(comp_i);
    mySolution->setSolutionFunction(ui_hat, ui_exact);

    VarPtr tn_i = this->tn_hat(comp_i);
    TFunctionPtr<double> tn_i_exact = p_exact * n->spatialComponent(comp_i);

    double sigma_weight = _stokesForm->mu();
    for (int comp_j=1; comp_j <= spaceDim; comp_j++)
    {
      VarPtr sigma_ij = this->sigma(comp_i, comp_j);
      TFunctionPtr<double> sigma_ij_exact = sigma_weight * ui_exact->grad(spaceDim)->spatialComponent(comp_j);
      mySolution->setSolutionFunction(sigma_ij, sigma_ij_exact);

      tn_i_exact = tn_i_exact - sigma_ij_exact * n->spatialComponent(comp_j);
      if (_conservationFormulation)
      {
        FunctionPtr uj_exact = u_exact->spatialComponent(comp_j);
        tn_i_exact = tn_i_exact + ui_exact*uj_exact*n->spatialComponent(comp_j);
      }
    }
    mySolution->setSolutionFunction(tn_i, tn_i_exact * sideParity);
  }

  return mySolution;
}

TFunctionPtr<double> OldroydBFormulation::forcingFunction(int spaceDim, double Re, TFunctionPtr<double> u_exact, TFunctionPtr<double> p_exact)
{
  FunctionPtr convectiveTerm = OldroydBFormulation::convectiveTerm(spaceDim, u_exact);
  return _stokesForm->forcingFunction(u_exact, p_exact) + convectiveTerm;
}

void OldroydBFormulation::setForcingFunction(FunctionPtr forcingFunction)
{
  // set the RHS:
  if (forcingFunction == Teuchos::null)
  {
    FunctionPtr scalarZero = Function::zero();
    if (_spaceDim == 1)
      forcingFunction = scalarZero;
    else if (_spaceDim == 2)
      forcingFunction = Function::vectorize(scalarZero, scalarZero);
    else if (_spaceDim == 3)
      forcingFunction = Function::vectorize(scalarZero, scalarZero, scalarZero);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space dimension");
  }

  _rhsForSolve = this->rhs(forcingFunction, _neglectFluxesOnRHS);
  _rhsForResidual = this->rhs(forcingFunction, false);
  _solnIncrement->setRHS(_rhsForSolve);
}

// ! returns the forcing function for steady-state Navier-Stokes corresponding to the indicated exact solution
TFunctionPtr<double> OldroydBFormulation::forcingFunctionSteady(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p)
{
  bool useConformingTraces = false; // doesn't matter for this
  StokesVGPFormulation stokesForm = StokesVGPFormulation::steadyFormulation(spaceDim, 1.0 / Re, useConformingTraces);

  return stokesForm.forcingFunction(u, p) + OldroydBFormulation::convectiveTerm(spaceDim, u);
}

// ! returns the forcing function for space-time Navier-Stokes corresponding to the indicated exact solution
TFunctionPtr<double> OldroydBFormulation::forcingFunctionSpaceTime(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p)
{
  bool useConformingTraces = false; // doesn't matter for this
  StokesVGPFormulation stokesForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, 1.0 / Re, useConformingTraces);

  return stokesForm.forcingFunction(u, p) + OldroydBFormulation::convectiveTerm(spaceDim, u);
}

double OldroydBFormulation::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double OldroydBFormulation::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

int OldroydBFormulation::nonlinearIterationCount()
{
  return _nonlinearIterationCount;
}

VarPtr OldroydBFormulation::p()
{
  return _stokesForm->p();
}

void OldroydBFormulation::setIP(IPPtr ip)
{
  _solnIncrement->setIP(ip);
}

RefinementStrategyPtr OldroydBFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void OldroydBFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void OldroydBFormulation::refine(RefinementStrategyPtr refStrategy)
{
  _nonlinearIterationCount = 0;
  _solnIncrement->setRHS(_rhsForResidual);
  refStrategy->refine();
  _solnIncrement->setRHS(_rhsForSolve);
}

void OldroydBFormulation::refine()
{
  _nonlinearIterationCount = 0;
  _solnIncrement->setRHS(_rhsForResidual);
  _refinementStrategy->refine();
  _solnIncrement->setRHS(_rhsForSolve);
}

void OldroydBFormulation::hRefine()
{
  _hRefinementStrategy->refine();
}

void OldroydBFormulation::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr OldroydBFormulation::rhs(TFunctionPtr<double> f, bool excludeFluxesAndTraces)
{
  int spaceDim = _stokesForm->spaceDim();

  // set the RHS:
  RHSPtr rhs = RHS::rhs();

  VarPtr v1 = this->v(1);
  VarPtr v2 = this->v(2);
  VarPtr v3;
  if (spaceDim==3) v3 = this->v(3);

  if (f != Teuchos::null)
  {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }

  TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);
  // subtract the stokesBF from the RHS:
  rhs->addTerm( -_stokesForm->bf()->testFunctional(backgroundFlowWeakReference, excludeFluxesAndTraces) );

  // finally, add the u sigma term:
  double Re = 1.0 / _stokesForm->mu();
  if (!_conservationFormulation)
  {
    for (int comp_i=1; comp_i <= spaceDim; comp_i++)
    {
      VarPtr vi = this->v(comp_i);

      for (int comp_j=1; comp_j <= spaceDim; comp_j++)
      {
        VarPtr uj = this->u(comp_j);
        TFunctionPtr<double> uj_prev = TFunction<double>::solution(uj,backgroundFlowWeakReference);
        VarPtr sigma_ij = this->sigma(comp_i, comp_j);
        TFunctionPtr<double> sigma_ij_prev = TFunction<double>::solution(sigma_ij, backgroundFlowWeakReference);
        rhs->addTerm((-Re * uj_prev * sigma_ij_prev) * vi);
      }
    }
  }
  else
  {
    if (spaceDim == 2)
    {
      VarPtr u1 = this->u(1);
      VarPtr u2 = this->u(2);
      VarPtr v1 = this->v(1);
      VarPtr v2 = this->v(2);
      TFunctionPtr<double> u1_prev = TFunction<double>::solution(u1,backgroundFlowWeakReference);
      TFunctionPtr<double> u2_prev = TFunction<double>::solution(u2,backgroundFlowWeakReference);
      rhs->addTerm( u1_prev * u1_prev * v1->dx() );
      rhs->addTerm( u1_prev * u2_prev * v1->dy() );
      rhs->addTerm( u2_prev * u1_prev * v2->dx() );
      rhs->addTerm( u2_prev * u2_prev * v2->dy() );
    }
    else if (spaceDim == 3)
    {
      VarPtr u1 = this->u(1);
      VarPtr u2 = this->u(2);
      VarPtr u3 = this->u(3);
      VarPtr v1 = this->v(1);
      VarPtr v2 = this->v(2);
      VarPtr v3 = this->v(3);
      TFunctionPtr<double> u1_prev = TFunction<double>::solution(u1,backgroundFlowWeakReference);
      TFunctionPtr<double> u2_prev = TFunction<double>::solution(u2,backgroundFlowWeakReference);
      TFunctionPtr<double> u3_prev = TFunction<double>::solution(u3,backgroundFlowWeakReference);
      rhs->addTerm( u1_prev * u1_prev * v1->dx() );
      rhs->addTerm( u1_prev * u2_prev * v1->dy() );
      rhs->addTerm( u1_prev * u3_prev * v1->dz() );

      rhs->addTerm( u2_prev * u1_prev * v2->dx() );
      rhs->addTerm( u2_prev * u2_prev * v2->dy() );
      rhs->addTerm( u2_prev * u3_prev * v2->dz() );

      rhs->addTerm( u3_prev * u1_prev * v3->dx() );
      rhs->addTerm( u3_prev * u2_prev * v3->dy() );
      rhs->addTerm( u3_prev * u3_prev * v3->dz() );
    }
  }
  // cout << endl <<endl << rhs->linearTerm()->displayString() << endl;

  return rhs;
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void OldroydBFormulation::save(std::string prefixString)
{
  _backgroundFlow->mesh()->saveToHDF5(prefixString+".mesh");
  _backgroundFlow->saveToHDF5(prefixString+".soln");
  _solnIncrement->saveToHDF5(prefixString+"_previous.soln");

  if (_streamSolution != Teuchos::null)
  {
    _streamSolution->mesh()->saveToHDF5(prefixString+"_stream.mesh");
    _streamSolution->saveToHDF5(prefixString + "_stream.soln");
  }
}

TSolutionPtr<double> OldroydBFormulation::solution()
{
  return _backgroundFlow;
}

TSolutionPtr<double> OldroydBFormulation::solutionIncrement()
{
  return _solnIncrement;
}

void OldroydBFormulation::solveAndAccumulate(double weight)
{
  RHSPtr savedRHS = _solnIncrement->rhs();
  _solnIncrement->setRHS(_rhsForSolve);
  _solnIncrement->solve(_solver);
  _solnIncrement->setRHS(savedRHS);

  bool allowEmptyCells = false;
  _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  _nonlinearIterationCount++;
}

VarPtr OldroydBFormulation::streamPhi()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamFormulation->phi();
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

TSolutionPtr<double> OldroydBFormulation::streamSolution()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamSolution;
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

void OldroydBFormulation::setSolver(SolverPtr solver)
{
  _solver = solver;
}

void OldroydBFormulation::setTimeStep(double dt)
{
  _stokesForm->setTimeStep(dt);
}

VarPtr OldroydBFormulation::sigma(int i, int j)
{
  return _stokesForm->sigma(i,j);
}

BFPtr OldroydBFormulation::stokesBF()
{
  return _stokesForm->bf();
}

VarPtr OldroydBFormulation::u(int i)
{
  return _stokesForm->u(i);
}

// traces:
VarPtr OldroydBFormulation::tn_hat(int i)
{
  return _stokesForm->tn_hat(i);
}

VarPtr OldroydBFormulation::u_hat(int i)
{
  return _stokesForm->u_hat(i);
}

// test variables:
VarPtr OldroydBFormulation::tau(int i)
{
  return _stokesForm->tau(i);
}

VarPtr OldroydBFormulation::v(int i)
{
  return _stokesForm->v(i);
}
