#ifndef DPG_SOLUTION
#define DPG_SOLUTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  Solution.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

// Epetra includes
#include <Epetra_Map.h>

#include "Epetra_Comm.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "Epetra_Operator.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "BasisCache.h"
#include "DofInterpreter.h"
#include "ElementType.h"
#include "LocalStiffnessMatrixFilter.h"
#include "Narrator.h"
#include "Solver.h"

namespace Camellia
{
template <typename Scalar>
  class TSolution : public Narrator
{
private:
  int _cubatureEnrichmentDegree;
  
  // first index into _solutionForCellID corresponds to the index of the RHS (0 for primary, 1 for influence, at present)
  std::vector<std::map< GlobalIndexType, Intrepid::FieldContainer<Scalar> > > _solutionForCellID;
  std::map< GlobalIndexType, double > _energyErrorForCell; // this is *just* for the primary solution.  Use RieszRep if dealing with the influence function...

  map< GlobalIndexType, Intrepid::FieldContainer<double> > _residualForCell;
  std::map< GlobalIndexType, Intrepid::FieldContainer<double> > _errorRepresentationForCell;

  // evaluates the inversion of the RHS
  std::map< GlobalIndexType,Intrepid::FieldContainer<Scalar> > _rhsRepresentationForCell;

  MeshPtr _mesh;
  TBCPtr<Scalar> _bc;
  Teuchos::RCP<DofInterpreter> _dofInterpreter; // defaults to Mesh
  Teuchos::RCP<DofInterpreter> _oldDofInterpreter; // the one saved when we turn on condensed solve
  TBFPtr<Scalar> _bf;
  TRHSPtr<Scalar> _rhs;
  TIPPtr<Scalar> _ip;
  
  LinearTermPtr _goalOrientedRHS;  // trial space functional.
  
  Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;
  Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

  Teuchos::RCP<Epetra_CrsMatrix> _globalStiffMatrix;
  Teuchos::RCP<Epetra_FEVector> _rhsVector;
  Teuchos::RCP<Epetra_FEVector> _lhsVector;

  TMatrixPtr<Scalar> _globalStiffMatrix2;
  TVectorPtr<Scalar> _rhsVector2;
  TVectorPtr<Scalar> _lhsVector2;

  bool _residualsComputed;
  bool _energyErrorComputed;
  bool _rankLocalEnergyErrorComputed;
  int _warnAboutDiscontinuousBCs = 1;
  // the  values of this map have dimensions (numCells, numTrialDofs)

  void initialize();
  void integrateBasisFunctions(Intrepid::FieldContainer<GlobalIndexTypeToCast> &globalIndices,
                               Intrepid::FieldContainer<Scalar> &values, int trialID);
  void integrateBasisFunctions(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID);

  // statistics for the last solve:
  double _totalTimeLocalStiffness, _totalTimeGlobalAssembly, _totalTimeBCImposition, _totalTimeSolve, _totalTimeDistributeSolution;
  double _meanTimeLocalStiffness, _meanTimeGlobalAssembly, _meanTimeBCImposition, _meanTimeSolve, _meanTimeDistributeSolution;
  double _maxTimeLocalStiffness, _maxTimeGlobalAssembly, _maxTimeBCImposition, _maxTimeSolve, _maxTimeDistributeSolution;
  double _minTimeLocalStiffness, _minTimeGlobalAssembly, _minTimeBCImposition, _minTimeSolve, _minTimeDistributeSolution;
  double _totalTimeApplyJumpTerms, _meanTimeApplyJumpTerms, _maxTimeApplyJumpTerms, _minTimeApplyJumpTerms;

  bool _reportConditionNumber, _reportTimingResults;
  bool _saveMeshOnSolveError = true; // if there is a solve error, save the mesh to disk for potential analysis
  bool _writeMatrixToMatlabFile;
  bool _writeMatrixToMatrixMarketFile;
  bool _writeRHSToMatrixMarketFile;
  bool _zmcsAsRankOneUpdate;
  bool _zmcsAsLagrangeMultipliers;

  std::string _matrixFilePath;
  std::string _rhsFilePath;
  
  std::string _solutionIdentifier = ""; // used to allow Solutions to be presented distinctly to the user (e.g., current and previous time step)

  GlobalIndexType _bcDofCount; // global count of the number of BCs imposed during last call to imposeBCs().
  
  double _globalSystemConditionEstimate;

  double _zmcRho;

  static double conditionNumberEstimate( Epetra_LinearProblem & problem, int &errCode );

  void setGlobalSolutionFromCellLocalCoefficients();

  void gatherSolutionData(); // get all solution data onto every node (not what we should do in the end)
//protected:
  // Pretty sure this is cruft... commenting it out
//  Intrepid::FieldContainer<Scalar> solutionForElementTypeGlobal(ElementTypePtr elemType); // probably should be deprecated…
public:
  TSolution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
            TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
  // Deprecated constructor, use the one which explicitly passes in BF
  // Will eventually be removing BF reference from Mesh
  TSolution(MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
            TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
  TSolution(const TSolution &soln);
  virtual ~TSolution() {}

  const Intrepid::FieldContainer<Scalar>& allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports=true,
                                                                   int solutionNumber=0); // coefficients for all solution variables
  void setLocalCoefficientsForCell(GlobalIndexType cellID, const Intrepid::FieldContainer<Scalar> &coefficients, int solutionNumber);

  Teuchos::RCP<DofInterpreter> getDofInterpreter() const;
  void setDofInterpreter(Teuchos::RCP<DofInterpreter> dofInterpreter);

  Epetra_Map getPartitionMap();
  Epetra_Map getPartitionMap(PartitionIndexType rank, std::set<GlobalIndexType> &myGlobalIndicesSet,
                             GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm );

  MapPtr getPartitionMap2();
  MapPtr getPartitionMap2(PartitionIndexType rank, std::set<GlobalIndexType> &myGlobalIndicesSet,
                          GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Teuchos_CommPtr Comm );

  Epetra_MultiVector* getGlobalCoefficients();
  TVectorPtr<Scalar> getGlobalCoefficients2();

  bool cellHasCoefficientsAssigned(GlobalIndexType cellID, int solutionOrdinal);
  void clearComputedResiduals();

  bool getZMCsAsGlobalLagrange() const;
  void setZMCsAsGlobalLagrange(bool value); // should be set before call to initializeLHSVector(), initializeStiffnessAndLoad()

  // solve steps:
  void initializeLHSVector();
  void initializeLoad(); // to be used if setStiffnessMatrix() will be called...
  void initializeStiffnessAndLoad();
  void applyDGJumpTerms();
  void populateStiffnessAndLoad();
  void imposeBCs();
  void imposeZMCsUsingLagrange(); // if not using Lagrange for ZMCs, puts 1's in the diagonal for these rows
  void setProblem(TSolverPtr<Scalar> solver);
  int solveWithPrepopulatedStiffnessAndLoad(TSolverPtr<Scalar> solver, bool callResolveInstead = false);
  void importSolution(); // imports for all rank-local cellIDs
  void importSolutionForOffRankCells(std::set<GlobalIndexType> cellIDs);
  void importGlobalSolution(); // imports (and interprets!) global solution.  NOT scalable.

  int solve();

  int solve(bool useMumps);

  int solve( TSolverPtr<Scalar> solver );

  void addSolution(TSolutionPtr<Scalar> soln, double weight, bool allowEmptyCells = false, bool replaceBoundaryTerms=false); // thisSoln += weight * soln

  void addSolution(TSolutionPtr<Scalar> soln, double weight, set<int> varsToAdd, bool allowEmptyCells = false); // thisSoln += weight * soln
  // will add terms in varsToAdd, but will replace all other variables
  void addReplaceSolution(TSolutionPtr<Scalar> soln, double weight, set<int> varsToAdd, set<int> varsToReplace, bool allowEmptyCells = false); // thisSoln += weight * soln
  
  // adds terms in varsToAdd using newValue = myWeight * oldValue + addedVarWeight * solnValue; replaces varsToReplace as newValue = solnValue
  void addReplaceSolution(TSolutionPtr<Scalar> soln, double addedVarWeight, double myWeight, set<int> varsToAdd, set<int> varsToReplace, bool allowEmptyCells = false); // thisSoln += weight * soln

  // static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
  // and returns a set of weights for the appropriate basis
  static void basisCoeffsForTrialOrder(Intrepid::FieldContainer<Scalar> &basisCoeffs, DofOrderingPtr trialOrder,
                                       const Intrepid::FieldContainer<Scalar> &allCoeffs, int trialID, int sideIndex);

  void clear();

  // ! After a problem has been set up (stiffness matrix, rhs assembled; BCs imposed), this method will compute and return a condition number estimate using AztecOO.
  double conditionNumberEstimate(int &errCode) const;
  
  int cubatureEnrichmentDegree() const;
  void setCubatureEnrichmentDegree(int value);
  
  const std::string & getIdentifier() const;
  void setIdentifier(const std::string &solutionIdentifier);
  // ! returns the global count of the number of degrees of freedom on which boundary conditions were imposed in
  // ! the last call to imposeBCs().  (imposeBCs() is called in solve().)
  GlobalIndexType getBCDofCount() const;

  void setSolution(TSolutionPtr<Scalar> soln); // thisSoln = soln

  void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID,
                      const Intrepid::FieldContainer<double> &physicalPoints, int solutionOrdinal=0); // searches for the elements that match the points provided
  void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache,
                      bool weightForCubature = false, Camellia::EOperator op = OP_VALUE, int solutionOrdinal=0);

  void solnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex=VOLUME_INTERIOR_SIDE_ORDINAL,
                           int solutionOrdinal=0);
  void setSolnCoeffsForCellID(const Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int trialID,
                              int sideIndex, int solutionOrdinal);
  void setSolnCoeffsForCellID(const Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int solutionOrdinal);

  const std::vector< std::map< GlobalIndexType, Intrepid::FieldContainer<Scalar> > > & solutionForCellID() const;

  double meshMeasure();

  double InfNormOfSolution(int trialID);
  double InfNormOfSolutionGlobal(int trialID);

  double L2NormOfSolution(int trialID);
  double L2NormOfSolutionGlobal(int trialID);

  Teuchos::RCP<LagrangeConstraints> lagrangeConstraints() const;

  // ! provides the GID (in stiffness/load) corresponding to an element lagrange constraint
  GlobalIndexType elementLagrangeIndex(GlobalIndexType cellID, int lagrangeOrdinal) const;
  
  // ! returns the number of solutions we're solving for; right now, this is 1 or 2 (2 in the case that goal-oriented RHS is set).
  int numSolutions() const;
  
  void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades);
  void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const std::set<GlobalIndexType> &cellIDsToSkip );

  void projectOntoMesh(const std::map<int, TFunctionPtr<Scalar> > &functionMap, int solutionOrdinal);
  void projectOntoCell(const std::map<int, TFunctionPtr<Scalar> > &functionMap, GlobalIndexType cellID, int sideIndex, int solutionOrdinal);
  void projectFieldVariablesOntoOtherSolution(TSolutionPtr<Scalar> otherSoln);

  void projectOldCellOntoNewCells(GlobalIndexType cellID, ElementTypePtr oldElemType,
                                  const vector<GlobalIndexType> &childIDs, int solutionOrdinal);
  void projectOldCellOntoNewCells(GlobalIndexType cellID, ElementTypePtr oldElemType,
                                  const Intrepid::FieldContainer<Scalar> &oldData,
                                  const std::vector<GlobalIndexType> &childIDs, int solutonOrdinal);
  void reverseParitiesForLocalCoefficients(GlobalIndexType cellID, const vector<int> &sidesWithChangedParities, int solutionOrdinal);

  void setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
  void setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter);
  void setReportConditionNumber(bool value);
  void setReportTimingResults(bool value);

  void computeResiduals();
  void computeErrorRepresentation();

  double globalCondEstLastSolve(); // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.

  void discardInactiveCellCoefficients();
  double energyErrorTotal();
  const map<GlobalIndexType,double> & rankLocalEnergyError();

  void setWriteMatrixToFile(bool value,const std::string &filePath);
  void setWriteMatrixToMatrixMarketFile(bool value,const std::string &filePath);
  void setWriteRHSToMatrixMarketFile(bool value, const std::string &filePath);

  MeshPtr mesh() const;
  TBFPtr<Scalar> bf() const;
  TBCPtr<Scalar> bc() const;
  TRHSPtr<Scalar> rhs() const;
  TIPPtr<Scalar> ip() const;
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter() const;

  void setBC( TBCPtr<Scalar> );
  void setRHS( TRHSPtr<Scalar> );
  
  // ! set an RHS involving trial-space variables.  Principal application at the moment
  // ! is for goal-oriented adaptivity (hence the name), but more generally this belongs to the DPG* method.
  void setGoalOrientedRHS( LinearTermPtr ltTrial );

  Teuchos::RCP<Epetra_CrsMatrix> getStiffnessMatrix();
  TMatrixPtr<Scalar> getStiffnessMatrix2();
  void setStiffnessMatrix(Teuchos::RCP<Epetra_CrsMatrix> stiffness);
  void setStiffnessMatrix2(TMatrixPtr<Scalar> stiffness);

  Teuchos::RCP<Epetra_FEVector> getRHSVector();
  Teuchos::RCP<Epetra_FEVector> getLHSVector();

  TVectorPtr<Scalar> getRHSVector2();
  TVectorPtr<Scalar> getLHSVector2();

  void setIP( TIPPtr<Scalar>);

#if defined(HAVE_MPI) && defined(HAVE_AMESOS_MUMPS)
  void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new MumpsSolver()), bool reduceMemoryFootprint = false,
                      std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>()); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#else
  void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new TAmesos2Solver<Scalar>()), bool reduceMemoryFootprint = false,
                      std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>()); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#endif
//  void readFromFile(const std::string &filePath, int solutionOrdinal);
//  void writeToFile(const std::string &filePath, int solutionOrdinal);

#ifdef HAVE_EPETRAEXT_HDF5
  void save(std::string meshAndSolutionPrefix);
  static TSolutionPtr<Scalar> load(TBFPtr<Scalar> bf, std::string meshAndSolutionPrefix);
  void saveToHDF5(std::string filename);
  void loadFromHDF5(std::string filename);
#endif

  // Default of 0 adapts the number of points based on poly order
  void writeToVTK(const std::string& filePath, unsigned int num1DPts=0);
  void writeFieldsToVTK(const std::string& filePath, unsigned int num1DPts=0);
  void writeTracesToVTK(const std::string& filePath);

  // statistics accessors:
  double totalTimeApplyJumpTerms();
  double totalTimeLocalStiffness();
  double totalTimeGlobalAssembly();
  double totalTimeBCImposition();
  double totalTimeSolve();
  double totalTimeDistributeSolution();

  double meanTimeApplyJumpTerms();
  double meanTimeLocalStiffness();
  double meanTimeGlobalAssembly();
  double meanTimeBCImposition();
  double meanTimeSolve();
  double meanTimeDistributeSolution();

  double maxTimeApplyJumpTerms();
  double maxTimeLocalStiffness();
  double maxTimeGlobalAssembly();
  double maxTimeBCImposition();
  double maxTimeSolve();
  double maxTimeDistributeSolution();

  double minTimeApplyJumpTerms();
  double minTimeLocalStiffness();
  double minTimeGlobalAssembly();
  double minTimeBCImposition();
  double minTimeSolve();
  double minTimeDistributeSolution();

  void reportTimings();

  // ! offRankCellsToInclude: the ones that the condensed dof interpreter needs to know how to interpret.  (Comes up in context of h-multigrid in particular.)
  void setUseCondensedSolve(bool value, std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>());

  bool usesCondensedSolve() const;
  
  void writeStatsToFile(const std::string &filePath, int precision=4);

  std::vector<int> getZeroMeanConstraints();
  void setZeroMeanConstraintRho(double value);
  double zeroMeanConstraintRho();

  /*
   0: don't warn
   1: warn, but don't print values (default)
   2: print values
   */
  int warnAboutDiscontinuousBCs() const;
  void setWarnAboutDiscontinuousBCs(int outputLevel);
  
  static TSolutionPtr<Scalar> solution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
                                       TRHSPtr<Scalar> rhs = Teuchos::null,
                                       TIPPtr<Scalar> ip = Teuchos::null);
  // Deprecated method, use the above one
  static TSolutionPtr<Scalar> solution(MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
                                       TRHSPtr<Scalar> rhs = Teuchos::null,
                                       TIPPtr<Scalar> ip = Teuchos::null);
};

extern template class TSolution<double>;
}


#endif
