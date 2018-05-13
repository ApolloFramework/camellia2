//
//  CompressibleNavierStokesProblem.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/15/18.
//
//  The idea here is just to separate out some of the problem-specific logic from CompressibleNavierStokes

#ifndef CompressibleNavierStokesProblem_hpp
#define CompressibleNavierStokesProblem_hpp

#include "TypeDefs.h"

namespace Camellia
{
  class CompressibleNavierStokesProblem
  {
  public:
    FunctionPtr rhoInitial();
    FunctionPtr TInitial();
    std::vector<FunctionPtr> uInitial();
    
    static Teuchos::RCP<CompressibleNavierStokesProblem> noh();
    static Teuchos::RCP<CompressibleNavierStokesProblem> rayleighTaylor();
    static Teuchos::RCP<CompressibleNavierStokesProblem> triplePoint();
    static Teuchos::RCP<CompressibleNavierStokesProblem> namedProblem(const std::string &problemName);
  };
}

#endif /* CompressibleNavierStokesProblem_hpp */
