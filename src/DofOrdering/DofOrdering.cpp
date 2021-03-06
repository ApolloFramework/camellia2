//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#include "DofOrdering.h"
#include "MultiBasis.h"
#include "BasisFactory.h"
#include "BasisReconciliation.h"

using namespace std;
using namespace Intrepid;
using namespace Camellia;

#define INDICES_INDEX(sideOrdinal) (sideOrdinal == VOLUME_INTERIOR_SIDE_ORDINAL) ? _volumeIndex : sideOrdinal

DofOrdering::DofOrdering(CellTopoPtr cellTopo)   // constructor
{
  _nextIndex = 0;
  _indexNeedsToBeRebuilt = false;
  _cellTopologyForSide[VOLUME_INTERIOR_SIDE_ORDINAL] = cellTopo;
  _indices.resize(cellTopo->getSideCount() + 1); // +1 for volume
  _volumeIndex = cellTopo->getSideCount();
}

void DofOrdering::addEntry(int varID, BasisPtr basis, int basisRank, int sideOrdinal)
{
  // test to see if we already have one matching this.  (If so, that's an error.)
  int sideIndex = INDICES_INDEX(sideOrdinal);
  if ( _indices[sideIndex].find(varID) != _indices[sideIndex].end() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION( true,
                                std::invalid_argument,
                                "Already have an entry in DofOrdering for this varID, sideIndex pair.");
  }
  else
  {
    _indices[sideIndex][varID] = vector<int>(basis->getCardinality());
  }

  vector<int>* dofIndices = &(_indices[sideIndex][varID]);

  for (vector<int>::iterator dofEntryIt = dofIndices->begin(); dofEntryIt != dofIndices->end(); dofEntryIt++)
  {
    *dofEntryIt = _nextIndex;
    _nextIndex++;
  }

  varIDs.insert(varID);

  _sidesForVarID[varID].push_back(sideOrdinal);

  numSidesForVarID[varID]++;
  //cout << "numSidesForVarID[" << varID << "]" << numSidesForVarID[varID] << endl;
  pair<int, int> basisKey = make_pair(varID,sideOrdinal);
//  _nextIndex += basis->getCardinality();
  bases[basisKey] = basis;
  basisRanks[varID] = basisRank;
}

void DofOrdering::addIdentification(int varID, int side1, int basisDofOrdinal1,
                                    int side2, int basisDofOrdinal2)
{
  _indexNeedsToBeRebuilt = true;
//  cout << "addIdentification: " << varID << ", (" << side1 << "," << basisDofOrdinal1 << ")=(" << side2 << "," << basisDofOrdinal2 << ")" << endl;
  pair<int, int> sidePair1; // defined so that sidePair1.side < sidePair2.side
  pair<int, int> sidePair2;
  if (side1 < side2)
  {
    sidePair1 = make_pair(side1,basisDofOrdinal1);
    sidePair2 = make_pair(side2,basisDofOrdinal2);
  }
  else if (side2 < side1)
  {
    sidePair1 = make_pair(side2,basisDofOrdinal2);
    sidePair2 = make_pair(side1,basisDofOrdinal1);
  }
  else     // side2 == side1 -- probably an exception
  {
    TEUCHOS_TEST_FOR_EXCEPTION( ( side1 == side2 ),
                                std::invalid_argument,
                                "addIdentification for side1==side2 not supported.");
  }
  dofIdentifications[make_pair(varID,sidePair2)] = sidePair1;
}

CellTopoPtr DofOrdering::cellTopology(int sideIndex) const
{
  return _cellTopologyForSide.find(sideIndex)->second;
}

void DofOrdering::copyLikeCoefficients( FieldContainer<double> &newValues, Teuchos::RCP<DofOrdering> oldDofOrdering,
                                        const FieldContainer<double> &oldValues )
{
  // copy the coefficients for the bases that agree between the two DofOrderings
  // requires that "like" bases are actually pointers to the same memory location
  TEUCHOS_TEST_FOR_EXCEPTION( newValues.rank() != 1, std::invalid_argument, "newValues.rank() != 1");
  TEUCHOS_TEST_FOR_EXCEPTION( newValues.size() != totalDofs(), std::invalid_argument, "newValues.size() != totalDofs()");
  TEUCHOS_TEST_FOR_EXCEPTION( oldValues.rank() != 1, std::invalid_argument, "oldValues.rank() != 1");
  TEUCHOS_TEST_FOR_EXCEPTION( oldValues.size() != oldDofOrdering->totalDofs(), std::invalid_argument, "oldValues.size() != oldDofOrdering->totalDofs()");

  newValues.initialize(0.0);

  for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++)
  {
    int varID = *varIDIt;
    // skip any varIDs that one or the other dofOrdering does not have:
    if (! oldDofOrdering->hasEntryForVarID(varID)) continue;
    if (! this->hasEntryForVarID(varID)) continue;

    const vector<int>* sides = &oldDofOrdering->getSidesForVarID(varID);
    for (int sideOrdinal : *sides)
    {
      // these two checks *should* be unnecessary; we can probably remove them.
//      if (!hasBasisEntry(varID, sideOrdinal)) continue;
//      if (!oldDofOrdering->hasBasisEntry(varID, sideOrdinal)) continue;
      BasisPtr basis = getBasis(varID,sideOrdinal);
      if (basis.get() == oldDofOrdering->getBasis(varID,sideOrdinal).get() )
      {
        // bases alike: copy coefficients
        int cardinality = basis->getCardinality();
        for (int dofOrdinal=0; dofOrdinal < cardinality; dofOrdinal++)
        {
          int dofIndex = getDofIndex(varID,dofOrdinal,sideOrdinal);
          newValues(dofIndex) = oldValues( oldDofOrdering->getDofIndex(varID,dofOrdinal,sideOrdinal) );
        }
      }
    }
  }
}

BasisPtr DofOrdering::getBasis(int varID, int sideIndex) const
{
  pair<int,int> key = make_pair(varID,sideIndex);
  map< pair<int,int>, BasisPtr >::const_iterator entry = bases.find(key);
  if (entry == bases.end())
  {
    //cout << *this;
    cout << "basis not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basis not found.");
  }
  return (*entry).second;
}

void DofOrdering::getDofCoords(Intrepid::FieldContainer<double> &dofCoords, int varID, int sideOrdinal)
{
  CellTopoPtr volumeTopo = _cellTopologyForSide.find(VOLUME_INTERIOR_SIDE_ORDINAL)->second;
  bool createSideCache = (sideOrdinal != VOLUME_INTERIOR_SIDE_ORDINAL);
  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache); // 1: faux cubature degree; true: also create side cache
  BasisPtr basis = getBasis(varID, sideOrdinal);
  int basisCardinality = basis->getCardinality();
  int spaceDim = volumeTopo->getDimension();
  
  dofCoords.resize(basisCardinality, spaceDim);
  if (sideOrdinal == VOLUME_INTERIOR_SIDE_ORDINAL)
  {
    basis->getDofCoords(dofCoords);
  }
  else
  {
    BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
    FieldContainer<double> sideDofCoords(basisCardinality, basis->domainTopology()->getDimension());
    basis->getDofCoords(sideDofCoords);
    sideCache->setRefCellPoints(sideDofCoords);
    const Intrepid::FieldContainer<double>* volumeCoords = &sideCache->getPhysicalCubaturePoints();
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        dofCoords(basisOrdinal,d) = (*volumeCoords)(0,basisOrdinal,d);
      }
    }
  }
}

//
void DofOrdering::getDofCoords(const Intrepid::FieldContainer<double> &physicalCellNodes,
                               Intrepid::FieldContainer<double> &dofCoords, int varID, int sideOrdinal)
{
  CellTopoPtr volumeTopo = _cellTopologyForSide.find(VOLUME_INTERIOR_SIDE_ORDINAL)->second;
  bool createSideCache = (sideOrdinal != VOLUME_INTERIOR_SIDE_ORDINAL);
  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1); // 1: faux cubature degree; true: also create side cache
  basisCache->setPhysicalCellNodes(physicalCellNodes, vector<GlobalIndexType>(), createSideCache);
  
  BasisPtr basis = getBasis(varID, sideOrdinal);
  int basisCardinality = basis->getCardinality();
  int spaceDim = volumeTopo->getDimension();
  
  if (sideOrdinal == VOLUME_INTERIOR_SIDE_ORDINAL)
  {
    FieldContainer<double> refSpaceDofCoords(basisCardinality, spaceDim);
    basis->getDofCoords(refSpaceDofCoords);
    basisCache->setRefCellPoints(refSpaceDofCoords);
    dofCoords = basisCache->getPhysicalCubaturePoints();
  }
  else
  {
    BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
    FieldContainer<double> sideDofCoords(basisCardinality, basis->domainTopology()->getDimension());
    basis->getDofCoords(sideDofCoords);
    sideCache->setRefCellPoints(sideDofCoords);
    dofCoords = sideCache->getPhysicalCubaturePoints();
  }
}

int DofOrdering::getDofIndex(int varID, int basisDofOrdinal, int sideOrdinal, int subSideIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION( ( _indexNeedsToBeRebuilt ),
                              std::invalid_argument,
                              "getDofIndex called when _indexNeedsToBeRebuilt = true.  Call rebuildIndex() first.");
  int sideIndex = INDICES_INDEX(sideOrdinal);
  
  if (subSideIndex >= 0)
  {
    // then we've got a MultiBasis, and the basisDofOrdinal we have is *relative* to the subbasis
    BasisPtr basis = getBasis(varID,sideIndex);
    if ( ! BasisFactory::basisFactory()->isMultiBasis(basis) )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subSideIndex >= 0 for non-MultiBasis...");
    }
    MultiBasis<>* multiBasis = (MultiBasis<>*) basis.get();
    //cout << "(basisDofOrdinal, subSideIndex) : (" << basisDofOrdinal << ", " << subSideIndex << ") --> ";
    basisDofOrdinal = multiBasis->relativeToAbsoluteDofOrdinal(basisDofOrdinal,subSideIndex);
    //cout << basisDofOrdinal << endl;
  }

  auto entryIt = _indices[sideIndex].find(varID);
  if ( entryIt != _indices[sideIndex].end() )
  {
    int dofIndex = ((*entryIt).second)[basisDofOrdinal];
    if ((dofIndex < 0) || (dofIndex >= _nextIndex))
    {
      cout << "dofIndex out of bounds.\n";
      TEUCHOS_TEST_FOR_EXCEPTION( (dofIndex < 0) || (dofIndex >= _nextIndex), std::invalid_argument, "dofIndex out of bounds.");
    }
    return dofIndex;
  }
  else
  {
    cout << "No entry found for dofIndex\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No entry found for DofIndex.");
    return -1;
  }
}

const vector<int> & DofOrdering::getDofIndices(int varID, int sideOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION( ( _indexNeedsToBeRebuilt ),
                              std::invalid_argument,
                              "getDofIndices called when _indexNeedsToBeRebuilt = true.  Call rebuildIndex() first.");

  int sideIndex = INDICES_INDEX(sideOrdinal);
  auto entryIt = _indices[sideIndex].find(varID);
  if ( entryIt == _indices[sideIndex].end() )
  {
    if (sideIndex == _volumeIndex)
    {
      cout << "No entry found for DofIndex " << varID << " in volume.\n";
    }
    else
    {
      cout << "No entry found for DofIndex " << varID << " on side " << sideIndex << endl;
    }
    cout << "DofOrdering has entries:\n";
    cout << *this;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No entry found for DofIndex.");
  }
  return (*entryIt).second;
}

int DofOrdering::getBasisCardinality(int varID, int sideIndex)
{
  BasisPtr basis = getBasis(varID,sideIndex);
  return basis->getCardinality();
}

int DofOrdering::getNumSidesForVarID(int varID)
{
  return numSidesForVarID[varID];
}

const vector<int> & DofOrdering::getSidesForVarID(int varID) const
{
  return _sidesForVarID.find(varID)->second;
}

int DofOrdering::getTotalBasisCardinality()   // total number of *distinct* basis functions
{
  int totalBasisCardinality = 0;
  set< ::Camellia::Basis<>* > basesCounted;
  for (map< pair<int, int>, BasisPtr >::iterator basisIt = bases.begin(); basisIt != bases.end(); basisIt++)
  {
    BasisPtr basis = basisIt->second;
    if (basesCounted.find(basis.get())==basesCounted.end())
    {
      basesCounted.insert(basis.get());
      totalBasisCardinality += basis->getCardinality();
    }
  }
  return totalBasisCardinality;
}

set<int> DofOrdering::getTraceDofIndices()
{
  // returns dof indices corresponding to variables with numSides > 1.
  set<int> traceDofIndices;
  for (int varID : varIDs)
  {
    const vector<int> *sides = &getSidesForVarID(varID);
    if (sides->size() > 1) // trace
    {
      for (int sideOrdinal : *sides)
      {
        const vector<int> * indices = &getDofIndices(varID, sideOrdinal);
        traceDofIndices.insert(indices->begin(),indices->end());
      }
    }
  }
  return traceDofIndices;
}

VarFactoryPtr DofOrdering::getVarFactory() const
{
  return _varFactory;
}

const set<int> & DofOrdering::getVarIDs() const
{
  return varIDs;
}

bool DofOrdering::hasBasisEntry(int varID, int sideIndex) const
{
  pair<int,int> key = make_pair(varID,sideIndex);
  map< pair<int,int>, BasisPtr >::const_iterator entry = bases.find(key);
  return entry != bases.end();
}

bool DofOrdering::hasEntryForVarID(int varID)
{
  return varIDs.find(varID) != varIDs.end();
}

bool DofOrdering::hasSideVarIDs()
{
  // returns true if there are any varIDs defined on more than one side
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
  {
    int varID = *varIt;
    if (numSidesForVarID[varID] > 1)
    {
      return true;
    }
  }
  return false;
}

bool DofOrdering::isTrialOrdering() const
{
  return _isTrialOrdering;
}

int DofOrdering::maxBasisDegree()
{
  map< pair<int,int>, BasisPtr >::iterator basisIterator;

  int maxBasisDegree = 0;

  for (basisIterator = bases.begin(); basisIterator != bases.end(); basisIterator++)
  {
    //pair< const pair<int,int>, BasisPtr > basisPair = *basisIterator;
    BasisPtr basis = (*basisIterator).second;
    if (maxBasisDegree < basis->getDegree() )
    {
      maxBasisDegree = basis->getDegree();
    }
  }
  return maxBasisDegree;
}

int DofOrdering::maxBasisDegreeForVolume()   // max degree among the varIDs with numSides == 1
{
  int maxBasisDegree = 0;

  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
  {
    int varID = *varIt;
    if (numSidesForVarID[varID]==1)
    {
      pair<int,int> key = make_pair(varID, VOLUME_INTERIOR_SIDE_ORDINAL);
      int degree = bases[key]->getDegree();
      if (maxBasisDegree < degree )
      {
        maxBasisDegree = degree;
      }
    }
  }
  return maxBasisDegree;
}

int DofOrdering::minimumSubcellDimensionForContinuity() const // across all bases
{
  CellTopoPtr volumeTopo = _cellTopologyForSide.find(VOLUME_INTERIOR_SIDE_ORDINAL)->second;
  int minSubcellDimension = volumeTopo->getDimension();
  
  for (auto entry : bases)
  {
    BasisPtr basis = entry.second;
    int basisMin = BasisReconciliation::minimumSubcellDimension(basis);
    if (minSubcellDimension > basisMin)
    {
      minSubcellDimension = basisMin;
    }
  }
  return minSubcellDimension;
}

void DofOrdering::print(std::ostream& os) const
{
  os << *this;
}

void DofOrdering::printDofCoords(int varID, std::ostream& os) const
{
  // define a lambda to create a point string:
  auto pointString = [] (int basisOrdinal, FieldContainer<double> &basisDofCoords) -> string
  {
    ostringstream pointStringStream;
    int spaceDim = basisDofCoords.dimension(1);
    pointStringStream << "(";
    for (int d=0; d < spaceDim - 1; d++)
    {
      pointStringStream << basisDofCoords(basisOrdinal, d) << ", ";
    }
    pointStringStream << basisDofCoords(basisOrdinal, spaceDim -1) << ")";
    
    return pointStringStream.str();
  };
  
  os << "******** Ref Space Dof Coords for var " << varID << ": *********\n";
  const vector<int>* sidesForVar = &this->getSidesForVarID(varID);
  
  CellTopoPtr volumeTopo = _cellTopologyForSide.find(VOLUME_INTERIOR_SIDE_ORDINAL)->second;
  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, true); // 1: faux cubature degree; true: also create side cache
  for (int sideOrdinal : *sidesForVar)
  {
    BasisPtr basis = getBasis(varID, sideOrdinal);
    const vector<int>* dofIndices = &getDofIndices(varID, sideOrdinal);
    
    FieldContainer<double> basisDofCoords(basis->getCardinality(), basis->domainTopology()->getDimension());
    if (sideOrdinal == VOLUME_INTERIOR_SIDE_ORDINAL)
    {
      os << "Volume dof ordinal:" << setw(20) << "Local Dof Index:" << setw(20) << "Coordinates:\n" << endl;
      for (int basisOrdinal=0; basisOrdinal < basis->getCardinality(); basisOrdinal++)
      {
        os << basisOrdinal << setw(20) << (*dofIndices)[basisOrdinal] << setw(20) << pointString(basisOrdinal, basisDofCoords) << endl;
      }
    }
    else
    {
      BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
      sideCache->setRefCellPoints(basisDofCoords);
      Intrepid::FieldContainer<double> basisDofCoordsVolume = sideCache->getPhysicalCubaturePoints();
      basisDofCoordsVolume.resize(basisDofCoordsVolume.dimension(1),basisDofCoordsVolume.dimension(2));
      
      os << "Side " << sideOrdinal << " dof ordinal:" << setw(20) << "Local Dof Index:" << setw(20) << "Coordinates:\n" << endl;
      for (int basisOrdinal=0; basisOrdinal < basis->getCardinality(); basisOrdinal++)
      {
        os << basisOrdinal << setw(20) << (*dofIndices)[basisOrdinal] << setw(20) << pointString(basisOrdinal, basisDofCoordsVolume) << endl;
      }
    }
  }
}


void DofOrdering::rebuildIndex()
{
  if (dofIdentifications.size() == 0)
  {
    // nothing to do
    return;
  }
  int numIdentificationsProcessed = 0;
  for (int varID : varIDs)
  {
    const vector<int>* sides = &getSidesForVarID(varID);
    for (int sideOrdinal : *sides)
    {
      BasisPtr basis = getBasis(varID,sideOrdinal);
      int cellTopoSideIndex = (numSidesForVarID[varID]==1) ? -1 : sideOrdinal;
      if ( _cellTopologyForSide.find(cellTopoSideIndex) == _cellTopologyForSide.end() )
      {
        CellTopoPtr cellTopoPtr = basis->domainTopology();
        _cellTopologyForSide[cellTopoSideIndex] = cellTopoPtr;
      }
      int sideIndex = INDICES_INDEX(sideOrdinal);
      for (int dofOrdinal=0; dofOrdinal < basis->getCardinality(); dofOrdinal++)
      {
        pair<int, pair<int,int> > key = make_pair(varID, make_pair(sideOrdinal,dofOrdinal));
        
        pair<int, int> indexKey = make_pair(varID,sideIndex); // key into indices container
        if ( dofIdentifications.find(key) != dofIdentifications.end() )
        {
          int earlierSideIndex  = INDICES_INDEX(dofIdentifications[key].first);
          int earlierDofOrdinal = dofIdentifications[key].second;
          if (_indices[sideIndex][varID][dofOrdinal] != _indices[earlierSideIndex][varID][earlierDofOrdinal])
          {
            _indices[sideIndex][varID][dofOrdinal] = _indices[earlierSideIndex][varID][earlierDofOrdinal];
//            cout << "processed identification for varID " << varID << ": (" << sideIndex << "," << dofOrdinal << ")";
//            cout << " --> " << "(" << earlierSideIndex << "," << earlierDofOrdinal << ")" << endl;
            numIdentificationsProcessed++;
          }
        }
        else
        {
          // modify the index according to the number of dofs we've consolidated
//          cout << "Reducing indices for key (varID=" << indexKey.first << ", sideIndex " << indexKey.second << ") for dofOrdinal " << dofOrdinal << " from ";
//          cout << indices[indexKey][dofOrdinal] << " to ";
          _indices[sideIndex][varID][dofOrdinal] -= numIdentificationsProcessed;
//          cout << indices[indexKey][dofOrdinal] << "\n";
        }
      }
    }
  }
  _nextIndex -= numIdentificationsProcessed;
  //cout << "index rebuilt; _nextIndex = " << _nextIndex << "; numIdentificationsProcessed: " << numIdentificationsProcessed << endl;
  _indexNeedsToBeRebuilt = false;
}

vector<pair<int,vector<int>>> DofOrdering::variablesWithNonZeroEntries(const Intrepid::FieldContainer<double> &localCoefficients, double tol) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(localCoefficients.size() != this->totalDofs(), std::invalid_argument, "localCoefficients must have size == totalDofs()");
  const set<int>* trialIDs = &this->getVarIDs();

  vector<pair<int,vector<int>>> results;
  for (int trialID : *trialIDs)
  {
    const vector<int>* trialSides = &this->getSidesForVarID(trialID);
    vector<int> matchingSides; // there may be several
    for (int trialSide : *trialSides)
    {
      const vector<int>* trialLocalDofIndices = &this->getDofIndices(trialID,trialSide);
      for (int trialLocalDofIndex : *trialLocalDofIndices)
      {
        if (abs(localCoefficients[trialLocalDofIndex]) > tol)
        {
          matchingSides.push_back(trialSide);
          break;
        }
      }
    }
    if (matchingSides.size() > 0)
    {
      results.push_back({trialID,matchingSides});
    }
  }
  return results;
}

void DofOrdering::setVarFactory(VarFactoryPtr varFactory, bool isTrialOrdering)
{
  _varFactory = varFactory;
  _isTrialOrdering = isTrialOrdering;
}

std::ostream& operator << (std::ostream& os, const DofOrdering& dofOrdering)
{
  // Save the format state of the original ostream os.
  Teuchos::oblackholestream oldFormatState;
  oldFormatState.copyfmt(os);

  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  os.setf(std::ios_base::right);

  VarFactoryPtr vf = dofOrdering.getVarFactory();
  bool isTrialOrdering = dofOrdering.isTrialOrdering();
  map< int, string > varNames;
  if (vf != Teuchos::null)
  {
    map<int, VarPtr> vars = (isTrialOrdering) ? vf->trialVars() : vf->testVars();
    for (auto entry : vars)
    {
      varNames[entry.first] = entry.second->name();
    }
  }
  
  set< int > varIDs = dofOrdering.getVarIDs();

  unsigned numVarIDs = varIDs.size();

  os<< "===============================================================================\n"\
    << "\t Number of varIDs = " << numVarIDs << "\n";
  os << "\t Number of dofs = " << dofOrdering.totalDofs() << endl;
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
  {
    int varID = *varIt;
    os << varID << " (" << dofOrdering.getSidesForVarID(varID).size() << " sides)";
    if (varNames.find(varID) != varNames.end())
    {
      os << " -- var name: " << varNames[varID];
    }
    os << endl;
  }

  if( numVarIDs == 0 )
  {
    os<< "====================================================================================\n"\
      << "|                        *** This is an empty DofOrdering ****                       |\n";
  }
  else
  {
    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
    {
      int varID = *varIt;
      const vector<int>* sides = &dofOrdering.getSidesForVarID(varID);
      for (int sideIndex : *sides)
      {
        if (! dofOrdering.hasBasisEntry(varID, sideIndex)) continue;
        os << "basis cardinality for varID=" << varID << ", side " << sideIndex << ": ";
        os << dofOrdering.getBasis(varID,sideIndex)->getCardinality() << endl;
        // TODO: output function space and/or the actual dofIndices for basis elements
      }
    }
    /*os<< "\t Dimensions     = ";

    for(int r = 0; r < rank; r++){
      os << " (" << dimensions[r] <<") ";
    }
    os << "\n";

    os<< "====================================================================================\n"\
    << "|              Multi-index          Enumeration             Value                  |\n"\
    << "====================================================================================\n";
    }

    for(int address = 0; address < numVarIDs; address++){
    container.getMultiIndex(multiIndex,address);
    std::ostringstream mistring;
    for(int r = 0; r < rank; r++){
      mistring <<  multiIndex[r] << std::dec << " ";
    }
    os.setf(std::ios::right, std::ios::adjustfield);
    os << std::setw(27) << mistring.str();
    os << std::setw(20) << address;
    os << "             ";
    os.setf(std::ios::left, std::ios::adjustfield);
    os << std::setw(myprec+8) << container[address] << "\n";
    }
     */

    os<< "====================================================================================\n\n";
  }
  // reset format state of os
  os.copyfmt(oldFormatState);

  return os;
}
