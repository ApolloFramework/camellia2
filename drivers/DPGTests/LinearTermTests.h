//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  LinearTermTests.h
//  Camellia
//

#ifndef Camellia_LinearTermTests_h
#define Camellia_LinearTermTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "LinearTerm.h"
#include "VarFactory.h"

#include "Mesh.h"

#include "BasisCache.h"

#include "BF.h"

class LinearTermTests : public TestSuite
{
  VarFactoryPtr varFactory;

  VarPtr v1, v2, v3; // HGRAD members (test variables)
  VarPtr q1, q2, q3; // HDIV members (test variables)
  VarPtr u1, u2, u3; // L2 members (trial variables)
  VarPtr u1_hat, u2_hat; // trace variables
  VarPtr u3_hat_n; // flux variable
  FunctionPtr sine_x, cos_y;
  Teuchos::RCP<Mesh> mesh;

  DofOrderingPtr trialOrder, testOrder;

  BasisCachePtr basisCache;

  BFPtr bf;

  void setup();
  void teardown();

  void transposeFieldContainer(FieldContainer<double> &fc);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testBoundaryPlusVolumeTerms();

  bool testSums();
  bool testIntegration();

  bool testRieszInversionAsProjection();
  bool testMixedTermConsistency();
  bool testRieszInversion();
  bool testIntegrateMixedBasis();

  bool testLinearTermEvaluation();

  std::string testSuiteName();
};



#endif
