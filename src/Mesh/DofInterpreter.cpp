//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
//
//  DofInterpreter.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/23/14.
//
//

#include "DofInterpreter.h"

#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"
#include "MPIWrapper.h"

#include "Epetra_Distributor.h"
#include "Teuchos_GlobalMPISession.hpp"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

set<GlobalIndexType> DofInterpreter::getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal)
{
  CellTopoPtr topo = _mesh->getElementType(cellID)->cellTopoPtr;
  int spaceDim = topo->getDimension();
  
  map<int, VarPtr> trialVars = _mesh->varFactory()->trialVars();
  
  VarPtr var = trialVars[varID];
  
  int subcellDim, subcellOrdinal;
  set<GlobalIndexType> fittableIndexSet;
  if ((var->varType() == FLUX) || (var->varType() == TRACE))
  {
    subcellDim = spaceDim - 1;
    subcellOrdinal = sideOrdinal;
  }
  else
  {
    subcellDim = spaceDim;
    subcellOrdinal = 0;
  }
  
  return this->globalDofIndicesForVarOnSubcell(varID, cellID, subcellDim, subcellOrdinal);
}

void DofInterpreter::interpretLocalCoefficients(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localCoefficients,
                                                map<GlobalIndexType,double> &fittedGlobalCoefficients, const set<int> &trialIDsToExclude)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  
  fittedGlobalCoefficients.clear();
  
  set<int> trialIDs = trialOrder->getVarIDs();
  for (int trialID : trialIDs)
  {
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    if (trialIDsToExclude.find(trialID) != trialIDsToExclude.end()) continue; // skip field
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      
      // check for non-zeros
      const vector<int>* localDofIndices = &trialOrder->getDofIndices(trialID, sideOrdinal);
      bool nonZeroFound = false;
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = (*localDofIndices)[basisOrdinal];
        if (localCoefficients[localDofIndex] != 0.0)
        {
          nonZeroFound = true;
          break;
        }
      }
      if (!nonZeroFound)
      {
        // then all zeros should be mapped; only question is which global dof indices get zeros
        std::set<GlobalIndexType> globalDofIndices = this->getGlobalDofIndices(cellID, trialID, sideOrdinal);
        for (GlobalIndexType globalDofIndex : globalDofIndices)
        {
          fittedGlobalCoefficients[globalDofIndex] = 0.0;
        }
        continue;
      }
      
      basisCoefficients.resize(basisCardinality);
      
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = (*localDofIndices)[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<double> fittedGlobalCoefficientsFC;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficientsFC,
                                      fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficientsFC.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        fittedGlobalCoefficients[globalDofIndex] = fittedGlobalCoefficientsFC[i];
      }
    }
  }
}

void DofInterpreter::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients,
                                                Epetra_MultiVector &globalCoefficients, int columnOrdinal)
{
  // TODO: make this method call the one above, which takes an STL map as argument.
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<double> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, columnOrdinal, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
        //        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

void DofInterpreter::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                                        FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData, FieldContainer<GlobalIndexType> &globalDofIndices)
{
  this->interpretLocalData(cellID,localStiffnessData,globalStiffnessData,globalDofIndices);
  FieldContainer<GlobalIndexType> globalDofIndicesForStiffness = globalDofIndices; // copy (for debugging/inspection purposes)
  this->interpretLocalData(cellID,localLoadData,globalLoadData,globalDofIndices);
  for (int i=0; i<globalDofIndicesForStiffness.size(); i++)
  {
    if (globalDofIndicesForStiffness[i] != globalDofIndices[i])
    {
      cout << "ERROR: the vector and matrix dof indices differ...\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: the vector and matrix dof indices differ...\n");
    }
  }
}

std::set<GlobalIndexType> DofInterpreter::importGlobalIndicesForCells(const std::vector<GlobalIndexType> &cellIDs)
{
  set<GlobalIndexType> dofIndicesSet;
  int rank = _mesh->Comm()->MyPID();

  // myRequestOwners should be in nondecreasing order (it appears)
  // this is accomplished by requestMap
  map<int, vector<GlobalIndexTypeToCast>> requestMap;
  
  for (GlobalIndexType cellID : cellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(cellID);
    if (partitionForCell == rank)
    {
      set<GlobalIndexType> dofIndicesForCell = this->globalDofIndicesForCell(cellID);
      dofIndicesSet.insert(dofIndicesForCell.begin(),dofIndicesForCell.end());
    }
    else
    {
      requestMap[partitionForCell].push_back(cellID);
    }
  }
  
  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;
  for (auto entry : requestMap)
  {
    int partition = entry.first;
    for (auto cellIDInPartition : entry.second)
    {
      myRequest.push_back(cellIDInPartition);
      myRequestOwners.push_back(partition);
    }
  }

  int myRequestCount = myRequest.size();

  Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*_mesh->Comm());

  GlobalIndexTypeToCast* myRequestPtr = NULL;
  int *myRequestOwnersPtr = NULL;
  if (myRequest.size() > 0)
  {
    myRequestPtr = &myRequest[0];
    myRequestOwnersPtr = &myRequestOwners[0];
  }
  int numCellsToExport = 0;
  GlobalIndexTypeToCast* cellIDsToExport = NULL;  // we are responsible for deleting the allocated arrays
  int* exportRecipients = NULL;

  distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numCellsToExport, cellIDsToExport, exportRecipients);

  const std::set<GlobalIndexType>* myCells = &_mesh->globalDofAssignment()->cellsInPartition(-1);

  vector<int> sizes(numCellsToExport);
  vector<GlobalIndexTypeToCast> indicesToExport;
  for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
    if (myCells->find(cellID) == myCells->end())
    {
      cout << "cellID " << cellID << " does not belong to rank " << rank << endl;
      ostringstream myRankDescriptor;
      myRankDescriptor << "rank " << rank << ", cellID ownership";
      Camellia::print(myRankDescriptor.str().c_str(), *myCells);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested cellID does not belong to this rank!");
    }

    set<GlobalIndexType> indicesForCell = this->globalDofIndicesForCell(cellID);
    indicesToExport.insert(indicesToExport.end(), indicesForCell.begin(), indicesForCell.end());
    sizes[cellOrdinal] = indicesForCell.size();
  }

  int objSize = sizeof(GlobalIndexTypeToCast) / sizeof(char);

  int importLength = 0;
  char* globalIndexData = NULL;
  int* sizePtr = NULL;
  char* indicesToExportPtr = NULL;
  if (numCellsToExport > 0)
  {
    sizePtr = &sizes[0];
    indicesToExportPtr = (char *) &indicesToExport[0];
  }
  distributor->Do(indicesToExportPtr, objSize, sizePtr, importLength, globalIndexData);
  const char* copyFromLocation = globalIndexData;
  int numDofsImport = importLength / objSize;
  vector<GlobalIndexTypeToCast> globalIndicesVector(numDofsImport);
  GlobalIndexTypeToCast* copyToLocation = &globalIndicesVector[0];
  for (int dofOrdinal=0; dofOrdinal<numDofsImport; dofOrdinal++)
  {
    memcpy(copyToLocation, copyFromLocation, objSize);
    copyFromLocation += objSize;
    copyToLocation++; // copyToLocation has type GlobalIndexTypeToCast*, so this moves the pointer by objSize bytes
  }

//  { // DEBUGGING
//    ostringstream myRankDescriptor;
//    myRankDescriptor << "rank " << rank << ", requested cells";
//    Camellia::print(myRankDescriptor.str().c_str(), cellIDs);
//
//    myRankDescriptor.str("");
//    myRankDescriptor << "rank " << rank << ", exported data";
//    Camellia::print(myRankDescriptor.str().c_str(), indicesToExport);
//
//    cout << "On rank " << rank << ", import length = " << importLength << endl;
//    myRankDescriptor.str("");
//    myRankDescriptor << "rank " << rank << ", imported data";
//    Camellia::print(myRankDescriptor.str().c_str(), globalIndicesVector);
//  }

  // debugging: introducing
  if( cellIDsToExport != 0 ) delete [] cellIDsToExport;
  if( exportRecipients != 0 ) delete [] exportRecipients;
  if (globalIndexData != 0 ) delete [] globalIndexData;

  dofIndicesSet.insert(globalIndicesVector.begin(),globalIndicesVector.end());

  return dofIndicesSet;
}

map<GlobalIndexType,set<GlobalIndexType>> DofInterpreter::importGlobalIndicesMap(const set<GlobalIndexType> &cellIDs)
{
  /*
   First implementation of this calls importGlobalIndicesForCells repeatedly (with max. one cell requested per rank).
   
   This is not optimal, but it's relatively easy to get right, and in initial use this method will not likely
   be a bottleneck.  (The better way would be to do it all in one go, and have a data format that indicates which
   global indices belong to which cells.  But that would be more effort to implement.)
   
   */
  
  Epetra_CommPtr Comm = _mesh->Comm();
  map<GlobalIndexType,set<GlobalIndexType>> globalIndicesMap;
  
  int myCount, maxGlobalCount; // always 0 or 1: how many cells we're asking for, whether anyone is still asking for a cell
  
  set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin();
  
  do {
    GlobalIndexType cellID;
    vector<GlobalIndexType> myRequest;
    if (cellIDIt != cellIDs.end())
    {
      myCount = 1;
      cellID = *cellIDIt++;
      myRequest = {cellID};
    }
    else
    {
      myCount = 0;
      myRequest = {};
    }
    Comm->MaxAll(&myCount, &maxGlobalCount, 1);
    
    if (maxGlobalCount > 0)
    {
      set<GlobalIndexType> globalIndicesForMyCell = this->importGlobalIndicesForCells(myRequest);
//      { // DEBUGGING
//        int rank = Teuchos::GlobalMPISession::getRank();
//        ostringstream label;
//        label << "rank " << rank << ", cell " << cellID << " globalDofIndices received";
//        print(label.str(), globalIndicesForMyCell);
//      }
      if (myCount > 0) globalIndicesMap[cellID] = globalIndicesForMyCell;
    }
  } while (maxGlobalCount > 0);
  
  return globalIndicesMap;
}
