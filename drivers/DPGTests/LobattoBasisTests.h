//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#ifndef Camellia_LobattoBasisTests_h
#define Camellia_LobattoBasisTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

class LobattoBasisTests : public TestSuite
{
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testLegendreValues();

  bool testLobattoValues();
  bool testLobattoDerivativeValues();

  bool testLobattoLineClassifications();
  bool testH1Classifications(); // checks that edge functions, vertex functions, etc. are correctly listed for the H^1 Lobatto basis

  bool testSimpleStiffnessMatrix();

  std::string testSuiteName();
};

#endif
