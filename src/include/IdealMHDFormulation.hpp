//
//  IdealMHDFormulation.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 5/1/18.
//

#ifndef IdealMHDFormulation_hpp
#define IdealMHDFormulation_hpp

#include "TypeDefs.h"

namespace Camellia
{
  class IdealMHDFormulation
  {
    BFPtr _steadyBF; // _bf with the time terms dropped, essentially
    BFPtr _bf;
    RHSPtr _rhs;
    
    int _spaceDim;
    bool _useConformingTraces;
    
    Teuchos::RCP<ParameterFunction> _fc, _fe; // forcing functions for continuity, energy equations
    std::vector<Teuchos::RCP<ParameterFunction> > _fm; // forcing for momentum equation(s)
    std::vector<Teuchos::RCP<ParameterFunction> > _fB; // forcing for magnetism equation(s)
    double _gamma;
    double _Pr;
    double _Cv;
    int _spatialPolyOrder;
    int _temporalPolyOrder;
    int _delta_k;
    std::string _filePrefix;
    double _time;
    bool _timeStepping;
    bool _spaceTime;
    double _t0; // used in space-time
    
    int _nonlinearIterationCount; // starts at 0, increases for each iterate
    
    Teuchos::RCP<ParameterFunction> _dt;
    Teuchos::RCP<ParameterFunction> _t;  // use a ParameterFunction so that user can easily "ramp up" BCs in time...
    
    SolverPtr _solver;
    
    int _solveCode;
    
    FunctionPtr _L2IncrementFunction, _L2SolutionFunction;
    
    // Abstract flux definitions (these are made concrete by evaluating at a Solution).
    FunctionPtr _massFlux;
    FunctionPtr _momentumFlux; // rank-2 tensor in 2D and 3D (_spaceDim rows, 3 columns).  In 1D, a 3-row vector.
    FunctionPtr _magneticFlux; // rank-2 tensor in 2D and 3D (_spaceDim rows, 3 columns).  In 1D, a 3-row vector.
    FunctionPtr _energyFlux;
    
    SolutionPtr _backgroundFlow, _solnIncrement, _solnPrevTime;
    
    RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;
    
    Teuchos::ParameterList _ctorParameters;
    
    std::map<int,int> _trialVariablePolyOrderAdjustments;
    
    VarFactoryPtr _vf;
    
    static const std::string S_rho;
    static const std::string S_m1, S_m2, S_m3;
    static const std::string S_E;
    static const std::string S_B1, S_B2, S_B3;
    
    static const std::string S_tc;
    static const std::string S_tm1, S_tm2, S_tm3;
    static const std::string S_te;
    static const std::string S_tB1, S_tB2, S_tB3;
    
    static const std::string S_vc;
    static const std::string S_vm1, S_vm2, S_vm3;
    static const std::string S_ve;
    static const std::string S_vB1, S_vB2, S_vB3;
    
    static const std::string S_m[3];
    static const std::string S_B[3];
    
    static const std::string S_tm[3];
    static const std::string S_tB[3];
    
    static const std::string S_vm[3];
    static const std::string S_vB[3];
    
    void CHECK_VALID_COMPONENT(int i); // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
    
    struct ScalarFluxEquation
    {
      VarPtr testVar;
      FunctionPtr flux;
      VarPtr traceVar;
      VarPtr timeTerm; // term that is differentiated in time
      FunctionPtr f_rhs;
    };
    
    std::map<int,ScalarFluxEquation> _fluxEquations; // keys are test var IDs
    
    FunctionPtr getMomentumFluxComponent(FunctionPtr momentumFlux, int i);
    
    FunctionPtr exactSolutionFlux(VarPtr testVar, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity);
  public:
    IdealMHDFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters);
    
    // ! the Ideal MHD formulation bilinear form
    BFPtr bf();
    
    // ! the Ideal MHD formulation rhs
    RHSPtr rhs();
    
    // ! the Ideal MHD formulation bilinear form, with any time terms dropped
    BFPtr steadyBF();
    
    void addMassFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B); // vector u, B
    void addMassFluxCondition(SpatialFilterPtr region, FunctionPtr value);
    
//    void addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr tm_i_exact, int i);
//    void addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact, int i);
    void addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    void addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr value);
    void addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
  
    void addMagneticFluxCondition(SpatialFilterPtr region, FunctionPtr value);
    void addMagneticFluxCondition(SpatialFilterPtr region, FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    // ! returns true if this is a space-time formulation; false otherwise.
    bool isSpaceTime() const;
    
    // ! returns true if this is a steady formulation; false otherwise.
    bool isSteady() const;
    
    // ! returns true if this is a time-stepping formulation; false otherwise.
    bool isTimeStepping() const;
    
    // ! declare inner product
    void setIP(IPPtr ip);
    
    void setIP( std::string normName );
    
    // ! returns the L^2 norm of the incremental solution
    double L2NormSolutionIncrement();
    
    // ! returns the L^2 norm of the background flow
    double L2NormSolution();
    
    // ! returns the nonlinear iteration count (since last refinement)
    int nonlinearIterationCount();
    
    // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
    // ! implementation, assumes that the constructor arguments provided to CompressibleNavierStokesFormulation were the same
    // ! on the CompressibleNavierStokesFormulation on which save() was invoked as they were for this CompressibleNavierStokesFormulation.
    void load(std::string prefixString);
  
    // ! Returns gamma
    double gamma();
    
    // ! Returns Pr
    double Pr();
    
    // ! Returns Cv
    double Cv();
    
    // ! Returns Cp
    double Cp();
    
    // ! Returns R
    double R();
    
    // ! refine according to energy error in the solution
    void refine();
    
    // ! h-refine according to energy error in the solution
    void hRefine();
    
    // ! p-refine according to energy error in the solution
    void pRefine();
    
    // ! returns the RefinementStrategy object being used to drive refinements
    RefinementStrategyPtr getRefinementStrategy();
    
    // ! Saves the solution(s) and mesh to an HDF5 format.
    void save(std::string prefixString);
    
    // ! set the RefinementStrategy to use for driving refinements
    void setRefinementStrategy(RefinementStrategyPtr refStrategy);
    
    // ! get the Solver used for the linear updates
    SolverPtr getSolver();
    
    // ! get the status of the last solve
    int getSolveCode();
    
    // ! set the Solver for the linear updates
    void setSolver(SolverPtr solver);
    
    // ! returns a FunctionPtr with the current timestep (dynamically updated)
    FunctionPtr getTimeStep();
    
    // ! set current time step used for transient solve
    void setTimeStep(double dt);
  
    // ! Returns the solution (at current time)
    SolutionPtr solution();
    
    // ! Returns the latest solution increment (at current time)
    SolutionPtr solutionIncrement();
    
    // ! Returns the solution (at previous time)
    SolutionPtr solutionPreviousTimeStep();
    
    // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
    double solveAndAccumulate();
    
    // ! Returns the L2 norm of the time residual
    double timeResidual();
    
    // ! Solves
    void solve();
    
    // ! Solves iteratively
    void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);
    
    // ! Returns the spatial dimension.
    int spaceDim();
    
    // field variables:
    VarPtr rho();           // density
    VarPtr m(int i);        // momentum
    VarPtr E();             // total energy
    VarPtr B(int i);        // magnetic field
    
    // traces:
    VarPtr tc();
    VarPtr tm(int i);
    VarPtr te();
    VarPtr tB();
    
    // test variables:
    VarPtr vc();
    VarPtr vm(int i);
    VarPtr ve();
    VarPtr vB(int i);
    
    // ! For an exact solution (u, rho, E, B), produces a map with solution variables that depend on them (fields, traces, fluxes).
    // ! If includeFluxParity is true, then fluxes includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeFluxParity is false, fluxes do not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    std::map<int, FunctionPtr> exactSolutionMap(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeFluxParity);
    
    // ! For an exact solution (u, rho, E, B), produces a map with volume solution variables that depend on them (fields).
    // ! If includeFluxParity is true, then fluxes includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    std::map<int, FunctionPtr> exactSolutionFieldMap(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding tc flux
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    FunctionPtr exactSolution_tc(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding tm flux.  (Since this is in general vector-valued, returns components in a std::vector.)
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    std::vector<FunctionPtr> exactSolution_tm(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding te flux
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    FunctionPtr exactSolution_te(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity);
  
    // ! For an exact solution (u, rho, E, B), returns the corresponding tB flux
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    FunctionPtr exactSolution_tB(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, bool includeParity);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding forcing in the continuity equation
    FunctionPtr exactSolution_fc(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding forcing in the momentum equation
    std::vector<FunctionPtr> exactSolution_fm(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    // ! For an exact solution (u, rho, E, B), returns the corresponding forcing in the energy equation
    FunctionPtr exactSolution_fe(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
  
    // ! For an exact solution (u, rho, E, B), returns the corresponding forcing in the magnetism equation
    std::vector<FunctionPtr> exactSolution_fB(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B);
    
    // ! returns a std::map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
    const std::map<int,int> &getTrialVariablePolyOrderAdjustments();
    
    // ! zeros out the solution increment
    void clearSolutionIncrement();
    
    Teuchos::ParameterList getConstructorParameters() const;
    
    // ! Set the forcing functions for problem.  f_momentum and f_magnetic should have components equal to the number of spatial dimensions
    void setForcing(FunctionPtr f_continuity, std::vector<FunctionPtr> f_momentum, FunctionPtr f_energy, std::vector<FunctionPtr> f_magnetic);
    
    // static utility functions:
    static Teuchos::RCP<IdealMHDFormulation> steadyFormulation(int spaceDim, MeshTopologyPtr meshTopo, int polyOrder, int delta_k);
    
    static Teuchos::RCP<IdealMHDFormulation> timeSteppingFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k);
    
    static Teuchos::RCP<IdealMHDFormulation> steadyEulerFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k);
    
    static Teuchos::RCP<IdealMHDFormulation> timeSteppingEulerFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k);
  }; // class IdealMHDFormulation
} // namespace Camellia

#endif /* IdealMHDFormulation_hpp */
