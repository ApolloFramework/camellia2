// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PardisoSolver.h
//  Camellia
//
//  Created by Brendan Keith Feb 2018
//
//

#ifndef Camellia_PardisoSolver_h
#define Camellia_PardisoSolver_h

#include "Amesos_config.h"
#include "Amesos.h"
#ifdef HAVE_AMESOS_PARDISO_MKL
#include "Amesos_Pardiso.h"

#include "CamelliaDebugUtility.h"
#include "MPIWrapper.h"

namespace Camellia {

  class PardisoSolver : public Solver
  {
    Teuchos::RCP<Amesos_Pardiso> _savedSolver;
    Teuchos::RCP<Epetra_LinearProblem> _savedProblem;
    bool _saveFactorization;
    bool _havePrintedStatus;
    bool _runSilent; // false by default -- true outputs when reuse is requested
    int _maxProcsToUse;
  public:
    PardisoSolver(bool saveFactorization)
    {
      _saveFactorization = saveFactorization;
      _havePrintedStatus = false;
      _runSilent = false;
      _maxProcsToUse = 256;
    }
    
    // ! Positive number will be the maximum number used.  -3 means all processors will be used; -2 means square root of the number of processors available will be used
    void setMaxProcsToUse(int maxProcs)
    {
      _maxProcsToUse = maxProcs;
    }
    
    int solve() {
      Teuchos::ParameterList paramList;
      
      int numRanks = this->_stiffnessMatrix->Comm().NumProc();
      int maxProcs = min(numRanks,_maxProcsToUse);
      paramList.set("MaxProcs",maxProcs); // -3 means all processors will be used; -2 means square root of the number of processors available will be used
      
//      if (!_havePrintedStatus)
//      {
//        if (MPIWrapper::rank() == 0)
//        {
//          cout << "Pardiso, about to solve with a matrix of " << this->_stiffnessMatrix->RowMap().NumGlobalElements() << " rows:\n";
//        }
//      }
      
      if (!_saveFactorization)
      {
        _havePrintedStatus = true;
        Epetra_LinearProblem problem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());
        Amesos_Pardiso pardiso(problem);
        pardiso.SetParameters(paramList);
        return pardiso.Solve();
      }
      else
      {
        _savedProblem = Teuchos::rcp( new Epetra_LinearProblem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get()) );
        _savedSolver = Teuchos::rcp( new Amesos_Pardiso(*_savedProblem) );
        
        // paramList.set("ReuseSymbolic",true);
        // paramList.set("Fact", "SamePattern_SameRowPerm");
        // _savedSolver->SetParameters(paramList);
        
        int err = _savedSolver->Solve();
        
        // if (!_havePrintedStatus && !_runSilent)
        // { // print status for diagnostic purposes
        //   if (MPIWrapper::rank() == 0) cout << "PardisoSolver.h : Pardiso factorization reuse requested.  Details of solver:\n";
        //   _savedSolver->PrintStatus();
        //   _havePrintedStatus = true;
        // }
        
        return err;
      }
    }
    
    int resolve() {
      if (_savedSolver.get() != NULL)
      {
        return _savedSolver->Solve();
      }
      else
      {
        return solve();
      }
    }
    
    void setRunSilent(bool value)
    {
      _runSilent = value;
    }
    
    virtual void stiffnessMatrixChanged()
    {
      _savedSolver = Teuchos::null;
      _savedProblem = Teuchos::null;
      _havePrintedStatus = false;
    }
  };
}
#endif

#endif
