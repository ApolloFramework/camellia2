//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MeshFactory.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CellDataMigration.h"
#include "GlobalDofAssignment.h"
#include "GnuPlotUtil.h"
#include "MOABReader.h"
#include "MPIWrapper.h"
#include "ParametricCurve.h"
#include "RefinementHistory.h"

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

using namespace Intrepid;
using namespace Camellia;

static ParametricCurvePtr parametricRect(double width, double height, double x0, double y0)
{
  // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle
  vector< pair<double, double> > vertices;
  vertices.push_back(make_pair(x0 + width/2.0, y0 + 0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 - height/2.0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 - height/2.0));
  return ParametricCurve::polygon(vertices);
}

map<int,int> MeshFactory::_emptyIntIntMap;

#ifdef HAVE_EPETRAEXT_HDF5
MeshPtr MeshFactory::loadFromHDF5(TBFPtr<double> bf, string filename, Epetra_CommPtr Comm)
{
  if (Comm == Teuchos::null)
  {
    Comm = MPIWrapper::CommWorld();
  }
  EpetraExt::HDF5 hdf5(*Comm);
  hdf5.Open(filename);
  
  MeshTopologyViewPtr meshTopoView = MeshTopologyView::readFromHDF5(Comm, hdf5);
  
  int numChunks;
  hdf5.Read("MeshTopology", "num chunks", numChunks);
  
  int myRank = Comm->MyPID();
  int numProcs = Comm->NumProc();
  vector<int> myChunkRanks; // ranks that were part of the write, now assigned to me
  if (numProcs < numChunks)
  {
    int chunksPerRank = numChunks / numProcs;
    int extraChunks = numChunks % numProcs;
    int myChunkCount;
    int myChunkStart;
    if (myRank < extraChunks)
    {
      myChunkCount = chunksPerRank + 1;
      myChunkStart = (chunksPerRank + 1) * myRank;
    }
    else
    {
      myChunkCount = chunksPerRank;
      myChunkStart = extraChunks + chunksPerRank * myRank;
    }
    for (int i=0; i<myChunkCount; i++)
    {
      myChunkRanks.push_back(myChunkStart + i);
    }
  }
  else
  {
    if (myRank < numChunks)
    {
      myChunkRanks.push_back(myRank);
    }
  }

  int trialOrderEnhancementsSize, testOrderEnhancementsSize, H1OrderSize;
  hdf5.Read("Mesh", "trialOrderEnhancementsSize", trialOrderEnhancementsSize);
  hdf5.Read("Mesh", "testOrderEnhancementsSize", testOrderEnhancementsSize);
  hdf5.Read("Mesh", "H1OrderSize", H1OrderSize);

  vector<int> trialOrderEnhancementsVec(trialOrderEnhancementsSize);
  vector<int> testOrderEnhancementsVec(testOrderEnhancementsSize);
  vector<int> H1Order(H1OrderSize);
  string GDARule;
  int deltaP;
  hdf5.Read("Mesh", "deltaP", deltaP);
  hdf5.Read("Mesh", "GDARule", GDARule);
  if (GDARule == "min")
  {
  }
  else if(GDARule == "max")
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "loadFromHDF5() does not currently supported maximum rule meshes");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid GDA");
  }
  if (trialOrderEnhancementsSize > 0)
  {
    hdf5.Read("Mesh", "trialOrderEnhancements", H5T_NATIVE_INT, trialOrderEnhancementsSize, &trialOrderEnhancementsVec[0]);
  }

  if (trialOrderEnhancementsSize > 0)
  {
    hdf5.Read("Mesh", "testOrderEnhancements", H5T_NATIVE_INT, testOrderEnhancementsSize, &testOrderEnhancementsVec[0]);
  }
  hdf5.Read("Mesh", "H1Order", H5T_NATIVE_INT, H1OrderSize, &H1Order[0]);

  vector<int> partitionCounts(numChunks);
  hdf5.Read("Mesh", "partition counts", H5T_NATIVE_INT, numChunks, &partitionCounts[0]);
  
  // we use the same assignments as above
  int myCellCount = 0;
  for (int myChunkRank : myChunkRanks)
  {
    myCellCount += partitionCounts[myChunkRank];
  }
  int globalActiveCellCount;
  Comm()->SumAll(&myCellCount, &globalActiveCellCount, 1);
  
  vector<int> myCellsIntVector(myCellCount); // may contain duplicates
  void* myCellsLocation = (myCellCount > 0) ? &myCellsIntVector[0] : NULL;
  hdf5.Read("Mesh", "partitions", myCellCount, globalActiveCellCount, H5T_NATIVE_INT, myCellsLocation);
  vector<GlobalIndexType> myCells(myCellsIntVector.begin(),myCellsIntVector.end());

  int pRefinementsSize; // in int entries (2 per p-refinement)
  hdf5.Read("Mesh", "p refinements size", pRefinementsSize);
  vector<int> pRefinementsVector(pRefinementsSize);
  if (pRefinementsSize > 0)
  {
    hdf5.Read("Mesh", "p refinements", H5T_NATIVE_INT, pRefinementsSize, &pRefinementsVector[0]);
  }
  map<GlobalIndexType,int> pRefinements;
  for (int i=0; i<pRefinementsSize/2; i++)
  {
    GlobalIndexType cellID = pRefinementsVector[i+0];
    int pRefinement = pRefinementsVector[i+1];
    pRefinements[cellID] = pRefinement;
  }

  hdf5.Close();

  map<int, int> trialOrderEnhancements;
  map<int, int> testOrderEnhancements;
  for (int i=0; i < trialOrderEnhancementsVec.size()/2; i++) // divide by two because we have 2 entries per var; map goes varID --> enhancement
  {
    trialOrderEnhancements[trialOrderEnhancementsVec[2*i]] = trialOrderEnhancementsVec[2*i+1];
  }
  for (int i=0; i < testOrderEnhancementsVec.size()/2; i++)
  {
    testOrderEnhancements[testOrderEnhancementsVec[2*i]] = testOrderEnhancementsVec[2*i+1];
  }

  // for now, we actually neglect the "myCells" container, and let that information come through the MeshTopology, if it is
  // distributed, or else through the usual default assignment from GlobalDofAssignment and the (default) partition policy.
  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopoView, bf, H1Order, deltaP, trialOrderEnhancements, testOrderEnhancements,
                                         Teuchos::null, Comm) );
  mesh->globalDofAssignment()->setCellPRefinements(pRefinements);
  return mesh;
}
// end HAVE_EPETRAEXT_HDF5 include guard
#endif

MeshPtr MeshFactory::minRuleMesh(MeshTopologyPtr meshTopo, TBFPtr<double> bf, int H1Order, int delta_k,
                                 Epetra_CommPtr Comm)
{
  std::vector<int> H1OrderVector(2);
  H1OrderVector[0] = H1Order;
  H1OrderVector[1] = H1Order;
  return minRuleMesh(meshTopo,bf,H1OrderVector,delta_k,Comm);
}

MeshPtr MeshFactory::minRuleMesh(MeshTopologyPtr meshTopo, TBFPtr<double> bf, vector<int> H1Order, int delta_k,
                                 Epetra_CommPtr Comm)
{
  std::map<int,int> emptyMap;
  MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopo, bf, H1Order, delta_k, emptyMap, emptyMap, nullPartitionPolicy, Comm) );
  mesh->enforceOneIrregularity();
  return mesh;
}

MeshPtr MeshFactory::quadMesh(Teuchos::ParameterList &parameters, Epetra_CommPtr Comm)
{
  bool useMinRule = parameters.get<bool>("useMinRule",true);
  TBFPtr<double> bf = parameters.get< TBFPtr<double> >("bf");
  int H1Order = parameters.get<int>("H1Order");
  int spaceDim = 2;
  int delta_k = parameters.get<int>("delta_k",spaceDim);
  double width = parameters.get<double>("width",1.0);
  double height = parameters.get<double>("height",1.0);
  int horizontalElements = parameters.get<int>("horizontalElements", 1);
  int verticalElements = parameters.get<int>("verticalElements", 1);
  bool divideIntoTriangles = parameters.get<bool>("divideIntoTriangles",false);
  double x0 = parameters.get<double>("x0",0.0);
  double y0 = parameters.get<double>("y0",0.0);
  map<int,int> emptyMap;
  map<int,int>* trialOrderEnhancements = parameters.get< map<int,int>* >("trialOrderEnhancements",&emptyMap);
  map<int,int>* testOrderEnhancements = parameters.get< map<int,int>* >("testOrderEnhancements",&emptyMap);
  vector< PeriodicBCPtr > emptyPeriodicBCs;
  vector< PeriodicBCPtr >* periodicBCs = parameters.get< vector< PeriodicBCPtr >* >("periodicBCs",&emptyPeriodicBCs);

  if (useMinRule)
  {
    MeshTopologyPtr meshTopology = quadMeshTopology(width,height,horizontalElements,verticalElements,divideIntoTriangles,x0,y0,*periodicBCs);
    return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, delta_k, *trialOrderEnhancements, *testOrderEnhancements,
                                  Teuchos::null, Comm) );
  }
  else
  {
    bool useConformingTraces = parameters.get<bool>("useConformingTraces", true);
//  cout << "periodicBCs size is " << periodicBCs->size() << endl;
    vector<vector<double> > vertices;
    vector< vector<IndexType> > allElementVertices;

    int numElements = divideIntoTriangles ? horizontalElements * verticalElements * 2 : horizontalElements * verticalElements;

    CellTopoPtr topo;
    if (divideIntoTriangles)
    {
      topo = Camellia::CellTopology::triangle();
    }
    else
    {
      topo = Camellia::CellTopology::quad();
    }
    vector< CellTopoPtr > cellTopos(numElements, topo);

    FieldContainer<double> quadBoundaryPoints(4,2);
    quadBoundaryPoints(0,0) = x0;
    quadBoundaryPoints(0,1) = y0;
    quadBoundaryPoints(1,0) = x0 + width;
    quadBoundaryPoints(1,1) = y0;
    quadBoundaryPoints(2,0) = x0 + width;
    quadBoundaryPoints(2,1) = y0 + height;
    quadBoundaryPoints(3,0) = x0;
    quadBoundaryPoints(3,1) = y0 + height;
    //  cout << "creating mesh with boundary points:\n" << quadBoundaryPoints;

    double southWest_x = quadBoundaryPoints(0,0),
           southWest_y = quadBoundaryPoints(0,1);

    double elemWidth = width / horizontalElements;
    double elemHeight = height / verticalElements;

    // set up vertices:
    // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
    vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
    for (int i=0; i<=horizontalElements; i++)
    {
      for (int j=0; j<=verticalElements; j++)
      {
        vertexIndices[i][j] = vertices.size();
        vector<double> vertex(spaceDim);
        vertex[0] = southWest_x + elemWidth*i;
        vertex[1] = southWest_y + elemHeight*j;
        vertices.push_back(vertex);
      }
    }

    for (int i=0; i<horizontalElements; i++)
    {
      for (int j=0; j<verticalElements; j++)
      {
        if (!divideIntoTriangles)
        {
          vector<IndexType> elemVertices;
          elemVertices.push_back(vertexIndices[i][j]);
          elemVertices.push_back(vertexIndices[i+1][j]);
          elemVertices.push_back(vertexIndices[i+1][j+1]);
          elemVertices.push_back(vertexIndices[i][j+1]);
          allElementVertices.push_back(elemVertices);
        }
        else
        {
          vector<IndexType> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
          elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
          elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
          elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
          elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
          elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
          elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH

          allElementVertices.push_back(elemVertices1);
          allElementVertices.push_back(elemVertices2);
        }
      }
    }

    return Teuchos::rcp( new Mesh(vertices, allElementVertices, bf, H1Order, delta_k, useConformingTraces, *trialOrderEnhancements, *testOrderEnhancements, *periodicBCs, Comm) );
  }
}

/*class ParametricRect : public ParametricCurve {
  double _width, _height, _x0, _y0;
  vector< ParametricCurvePtr > _edgeLines;
  vector< double > _switchValues;
public:
  ParametricRect(double width, double height, double x0, double y0) {
    // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle

    _width = width; _height = height; _x0 = x0; _y0 = y0;
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + 0, x0 + width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + height/2.0, x0 - width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 + height/2.0, x0 - width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 - height/2.0, x0 + width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 - height/2.0, x0 + width/2.0, y0 + 0));

    // switchValues are the points in (0,1) where we switch from one edge line to the next
    _switchValues.push_back(0.0);
    _switchValues.push_back(0.125);
    _switchValues.push_back(0.375);
    _switchValues.push_back(0.625);
    _switchValues.push_back(0.875);
    _switchValues.push_back(1.0);
  }
  void value(double t, double &x, double &y) {
    for (int i=0; i<_edgeLines.size(); i++) {
      if ( (t >= _switchValues[i]) && (t <= _switchValues[i+1]) ) {
        double edge_t = (t - _switchValues[i]) / (_switchValues[i+1] - _switchValues[i]);
        _edgeLines[i]->value(edge_t, x, y);
        return;
      }
    }
  }
};*/

MeshPtr MeshFactory::quadMesh(TBFPtr<double> bf, int H1Order, FieldContainer<double> &quadNodes, int pToAddTest,
                              Epetra_CommPtr Comm)
{
  if (quadNodes.size() != 8)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "quadNodes must be 4 x 2");
  }
  int spaceDim = 2;
  vector< vector<double> > vertices;
  for (int i=0; i<4; i++)
  {
    vector<double> vertex(spaceDim);
    vertex[0] = quadNodes[2*i];
    vertex[1] = quadNodes[2*i+1];
    vertices.push_back(vertex);
  }
  vector< vector<IndexType> > elementVertices;
  vector<IndexType> cell0;
  cell0.push_back(0);
  cell0.push_back(1);
  cell0.push_back(2);
  cell0.push_back(3);
  elementVertices.push_back(cell0);

  map<int,int> emptyMap;
  vector<PeriodicBCPtr> emptyPeriodicBCs;
  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bf, H1Order, pToAddTest, true, emptyMap, emptyMap,
                                        emptyPeriodicBCs, Comm) );
  return mesh;
}

MeshPtr MeshFactory::quadMesh(TBFPtr<double> bf, int H1Order, int pToAddTest,
                              double width, double height, int horizontalElements, int verticalElements, bool divideIntoTriangles,
                              double x0, double y0, vector<PeriodicBCPtr> periodicBCs,
                              Epetra_CommPtr Comm)
{

  Teuchos::ParameterList pl;

  pl.set("useMinRule", false);
  pl.set("bf",bf);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("width", width);
  pl.set("height", height);
  pl.set("divideIntoTriangles", divideIntoTriangles);
  pl.set("x0",x0);
  pl.set("y0",y0);
  pl.set("periodicBCs", &periodicBCs);

  return quadMesh(pl, Comm);
}

MeshPtr MeshFactory::quadMeshMinRule(TBFPtr<double> bf, int H1Order, int pToAddTest,
                                     double width, double height, int horizontalElements, int verticalElements,
                                     bool divideIntoTriangles, double x0, double y0, vector<PeriodicBCPtr> periodicBCs,
                                     Epetra_CommPtr Comm)
{
  Teuchos::ParameterList pl;

  pl.set("useMinRule", true);
  pl.set("bf",bf);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("width", width);
  pl.set("height", height);
  pl.set("divideIntoTriangles", divideIntoTriangles);
  pl.set("x0",x0);
  pl.set("y0",y0);
  pl.set("periodicBCs", &periodicBCs);

  return quadMesh(pl, Comm);
}

MeshTopologyPtr MeshFactory::quadMeshTopology(double width, double height, int horizontalElements, int verticalElements, bool divideIntoTriangles,
    double x0, double y0, vector<PeriodicBCPtr> periodicBCs)
{
  vector<vector<double> > vertices;
  vector< vector<IndexType> > allElementVertices;

  int numElements = divideIntoTriangles ? horizontalElements * verticalElements * 2 : horizontalElements * verticalElements;

  CellTopoPtr topo;
  if (divideIntoTriangles)
  {
    topo = Camellia::CellTopology::triangle();
  }
  else
  {
    topo = Camellia::CellTopology::quad();
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);

  int spaceDim = 2;

  FieldContainer<double> quadBoundaryPoints(4,spaceDim);
  quadBoundaryPoints(0,0) = x0;
  quadBoundaryPoints(0,1) = y0;
  quadBoundaryPoints(1,0) = x0 + width;
  quadBoundaryPoints(1,1) = y0;
  quadBoundaryPoints(2,0) = x0 + width;
  quadBoundaryPoints(2,1) = y0 + height;
  quadBoundaryPoints(3,0) = x0;
  quadBoundaryPoints(3,1) = y0 + height;
  //  cout << "creating mesh with boundary points:\n" << quadBoundaryPoints;

  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1);

  double elemWidth = width / horizontalElements;
  double elemHeight = height / verticalElements;

  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++)
  {
    for (int j=0; j<=verticalElements; j++)
    {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }

  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      if (!divideIntoTriangles)
      {
        vector<IndexType> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
      else
      {
        vector<IndexType> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);
        elemVertices1.push_back(vertexIndices[i+1][j]);
        elemVertices1.push_back(vertexIndices[i+1][j+1]);
        elemVertices2.push_back(vertexIndices[i][j+1]);
        elemVertices2.push_back(vertexIndices[i][j]);
        elemVertices2.push_back(vertexIndices[i+1][j+1]);

        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }

  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, allElementVertices, cellTopos));
  return Teuchos::rcp( new MeshTopology(geometry, periodicBCs) );
}

MeshPtr MeshFactory::hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                TBFPtr<double> bilinearForm, int H1Order, int pToAddTest)
{
  return shiftedHemkerMesh(-meshWidth/2, meshWidth/2, meshHeight, cylinderRadius, bilinearForm, H1Order, pToAddTest);
}

MeshTopologyPtr MeshFactory::importMOABMesh(string filePath, bool readInParallel)
{
  return MOABReader::readMOABMesh(filePath, !readInParallel);
}

MeshPtr MeshFactory::intervalMesh(TBFPtr<double> bf, double xLeft, double xRight, int numElements, int H1Order, int delta_k)
{
  MeshTopologyPtr meshTopology = intervalMeshTopology(xLeft, xRight, numElements);
  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, delta_k) );
}

MeshTopologyPtr MeshFactory::intervalMeshTopology(double xLeft, double xRight, int numElements, bool usePeriodicBCs)
{
  std::vector<PeriodicBCPtr> periodicBCs;
  if (usePeriodicBCs)
  {
    auto bc = PeriodicBC::xIdentification(xLeft, xRight);
    periodicBCs.push_back(bc);
  }
  
  return MeshFactory::intervalMeshTopology(xLeft, xRight, numElements, periodicBCs);
}

MeshTopologyPtr MeshFactory::intervalMeshTopology(double xLeft, double xRight, int numElements, vector<PeriodicBCPtr> periodicBCs)
{
  int n = numElements;
  vector< vector<double> > vertices(n+1);
  vector<double> vertex(1);
  double length = xRight - xLeft;
  vector< vector<IndexType> > elementVertices(n);
  vector<IndexType> oneElement(2);
  for (int i=0; i<n+1; i++)
  {
    vertex[0] = xLeft + (i * length) / n;
    //    cout << "vertex " << i << ": " << vertex[0] << endl;
    vertices[i] = vertex;
    if (i != n)
    {
      oneElement[0] = i;
      oneElement[1] = i+1;
      elementVertices[i] = oneElement;
    }
  }
  CellTopoPtr topo = Camellia::CellTopology::line();
  vector< CellTopoPtr > cellTopos(numElements, topo);
  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));
  
  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry, periodicBCs) );
  return meshTopology;
}

MeshPtr MeshFactory::rectilinearMesh(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts, int H1Order,
                                     int pToAddTest, vector<double> x0, map<int,int> trialOrderEnhancements,
                                     map<int,int> testOrderEnhancements, Epetra_CommPtr Comm)
{
  int spaceDim = dimensions.size();
  if (pToAddTest==-1)
  {
    pToAddTest = spaceDim;
  }

  MeshTopologyPtr meshTopology = rectilinearMeshTopology(dimensions, elementCounts, x0);

  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, pToAddTest, trialOrderEnhancements, testOrderEnhancements,
                                Teuchos::null, Comm) );
}

MeshTopologyPtr MeshFactory::rectilinearMeshTopology(vector<double> dimensions, vector<int> elementCounts, vector<double> x0,
                                                     vector<PeriodicBCPtr> periodicBCs)
{
  int spaceDim = dimensions.size();

  if (x0.size()==0)
  {
    for (int d=0; d<spaceDim; d++)
    {
      x0.push_back(0.0);
    }
  }

  if (elementCounts.size() != dimensions.size())
  {
    cout << "Element count container must match dimensions container in length.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Element count container must match dimensions container in length.\n");
  }

  if (spaceDim == 1)
  {
    double xLeft = x0[0];
    double xRight = dimensions[0] + xLeft;
    return MeshFactory::intervalMeshTopology(xLeft, xRight, elementCounts[0], periodicBCs);
  }

  if (spaceDim == 2)
  {
    return MeshFactory::quadMeshTopology(dimensions[0], dimensions[1], elementCounts[0], elementCounts[1], false, x0[0], x0[1], periodicBCs);
  }

  if (spaceDim != 3)
  {
    cout << "For now, only spaceDim 1,2,3 are supported by this MeshFactory method.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "For now, only spaceDim 1,2,3 are is supported by this MeshFactory method.");
  }

  CellTopoPtr topo;
  if (spaceDim==1)
  {
    topo = Camellia::CellTopology::line();
  }
  else if (spaceDim==2)
  {
    topo = Camellia::CellTopology::quad();
  }
  else if (spaceDim==3)
  {
    topo = Camellia::CellTopology::hexahedron();
  }
  else
  {
    cout << "Unsupported spatial dimension.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported spatial dimension");
  }

  int numElements = 1;
  vector<double> elemLinearMeasures(spaceDim);
  vector<double> origin = x0;
  for (int d=0; d<spaceDim; d++)
  {
    numElements *= elementCounts[d];
    elemLinearMeasures[d] = dimensions[d] / elementCounts[d];
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);

  map< vector<int>, IndexType> vertexLookup;
  vector< vector<double> > vertices;

  for (int i=0; i<elementCounts[0]+1; i++)
  {
    double x = origin[0] + elemLinearMeasures[0] * i;

    for (int j=0; j<elementCounts[1]+1; j++)
    {
      double y = origin[1] + elemLinearMeasures[1] * j;

      for (int k=0; k<elementCounts[2]+1; k++)
      {
        double z = origin[2] + elemLinearMeasures[2] * k;

        vector<int> vertexIndex;
        vertexIndex.push_back(i);
        vertexIndex.push_back(j);
        vertexIndex.push_back(k);

        vector<double> vertex;
        vertex.push_back(x);
        vertex.push_back(y);
        vertex.push_back(z);

        vertexLookup[vertexIndex] = vertices.size();
        vertices.push_back(vertex);
      }
    }
  }

  vector< vector<IndexType> > elementVertices;
  for (int i=0; i<elementCounts[0]; i++)
  {
    for (int j=0; j<elementCounts[1]; j++)
    {
      for (int k=0; k<elementCounts[2]; k++)
      {
        vector< vector<int> > vertexIntCoords(8, vector<int>(3));
        vertexIntCoords[0][0] = i;
        vertexIntCoords[0][1] = j;
        vertexIntCoords[0][2] = k;
        vertexIntCoords[1][0] = i+1;
        vertexIntCoords[1][1] = j;
        vertexIntCoords[1][2] = k;
        vertexIntCoords[2][0] = i+1;
        vertexIntCoords[2][1] = j+1;
        vertexIntCoords[2][2] = k;
        vertexIntCoords[3][0] = i;
        vertexIntCoords[3][1] = j+1;
        vertexIntCoords[3][2] = k;
        vertexIntCoords[4][0] = i;
        vertexIntCoords[4][1] = j;
        vertexIntCoords[4][2] = k+1;
        vertexIntCoords[5][0] = i+1;
        vertexIntCoords[5][1] = j;
        vertexIntCoords[5][2] = k+1;
        vertexIntCoords[6][0] = i+1;
        vertexIntCoords[6][1] = j+1;
        vertexIntCoords[6][2] = k+1;
        vertexIntCoords[7][0] = i;
        vertexIntCoords[7][1] = j+1;
        vertexIntCoords[7][2] = k+1;

        vector<IndexType> elementVertexOrdinals;
        for (int n=0; n<8; n++)
        {
          elementVertexOrdinals.push_back(vertexLookup[vertexIntCoords[n]]);
        }

        elementVertices.push_back(elementVertexOrdinals);
      }
    }
  }

  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry, periodicBCs) );
  return meshTopology;
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius)
{
  return shiftedHemkerGeometry(xLeft, xRight, -meshHeight/2.0, meshHeight/2.0, cylinderRadius);
}

MeshGeometryPtr MeshFactory::shiftedSquareCylinderGeometry(double xLeft, double xRight, double meshHeight, double squareDiameter)
{
  vector< vector<double> > vertices;
  vector<double> xs = {xLeft, -squareDiameter/2, squareDiameter/2, xRight};
  vector<double> ys = {-meshHeight/2, -squareDiameter/2, squareDiameter/2, meshHeight/2};
  for (int j=0; j < 4; j++)
  {
    for (int i=0; i < 4; i++)
    {
      vector<double> vertex(2);
      vertex[0] = xs[i];
      vertex[1] = ys[j];
      vertices.push_back(vertex);
    }
  }

  vector< vector<IndexType> > elementVertices;
  vector< CellTopoPtr > cellTopos;
  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  for (unsigned j=0; j < 3; j++)
  {
    for (unsigned i=0; i < 3; i++)
    {
      vector<IndexType> elVertex;
      elVertex.push_back(4*j+i);
      elVertex.push_back(4*j+i+1);
      elVertex.push_back(4*(j+1)+i+1);
      elVertex.push_back(4*(j+1)+i);
      if (!(i == 1 && j == 1))
      {
        elementVertices.push_back(elVertex);
        cellTopos.push_back(quad_4);
      }
    }
  }


  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
}

MeshPtr MeshFactory::readMesh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  ifstream mshFile;
  mshFile.open(filePath.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(mshFile.fail(), std::invalid_argument, "Could not open msh file");
  string line;
  getline(mshFile, line);
  while (line != "$Nodes")
  {
    getline(mshFile, line);
  }
  int numNodes;
  mshFile >> numNodes;
  vector<vector<double> > vertices;
  int dummy;
  for (int i=0; i < numNodes; i++)
  {
    vector<double> vertex(2);
    mshFile >> dummy;
    mshFile >> vertex[0] >> vertex[1] >> dummy;
    vertices.push_back(vertex);
  }
  while (line != "$Elements")
  {
    getline(mshFile, line);
  }
  int numElems;
  mshFile >> numElems;
  int elemType;
  int numTags;
  vector< vector<IndexType> > elementIndices;
  for (int i=0; i < numElems; i++)
  {
    mshFile >> dummy >> elemType >> numTags;
    for (int j=0; j < numTags; j++)
      mshFile >> dummy;
    if (elemType == 2)
    {
      vector<IndexType> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    if (elemType == 4)
    {
      vector<IndexType> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    else
    {
      getline(mshFile, line);
    }
  }
  mshFile.close();

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  ifstream nodeFile;
  ifstream eleFile;
  string nodeFileName = filePath+".node";
  string eleFileName = filePath+".ele";
  nodeFile.open(nodeFileName.c_str());
  eleFile.open(eleFileName.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(nodeFile.fail(), std::invalid_argument, "Could not open node file: "+nodeFileName);
  TEUCHOS_TEST_FOR_EXCEPTION(eleFile.fail(), std::invalid_argument, "Could not open ele file: "+eleFileName);
  // Read node file
  string line;
  int numNodes;
  nodeFile >> numNodes;
  getline(nodeFile, line);
  vector<vector<double> > vertices;
  int dummy;
  int spaceDim = 2;
  vector<double> pt(spaceDim);
  for (int i=0; i < numNodes; i++)
  {
    nodeFile >> dummy >> pt[0] >> pt[1];
    getline(nodeFile, line);
    vertices.push_back(pt);
  }
  nodeFile.close();
  // Read ele file
  int numElems;
  eleFile >> numElems;
  getline(eleFile, line);
  vector< vector<IndexType> > elementIndices;
  vector<IndexType> el(3);
  for (int i=0; i < numElems; i++)
  {
    eleFile >> dummy >> el[0] >> el[1] >> el[2];
    el[0]--;
    el[1]--;
    el[2]--;
    elementIndices.push_back(el);
  }
  eleFile.close();

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                                   int horizontalElements, int verticalElements,
                                   TBFPtr<double> bilinearForm,
                                   int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                                   map<int,int> trialOrderEnhancements,
                                   map<int,int> testOrderEnhancements)
{
  //  if (triangulate) cout << "Mesh: Triangulating\n" << endl;
  int pToAddTest = pTest - H1Order;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order

  // check that inputs match the assumptions (of a rectilinear mesh)
  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                              std::invalid_argument,
                              "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1),
         southEast_x = quadBoundaryPoints(1,0),
//  southEast_y = quadBoundaryPoints(1,1),
//  northEast_x = quadBoundaryPoints(2,0),
//  northEast_y = quadBoundaryPoints(2,1),
//  northWest_x = quadBoundaryPoints(3,0),
         northWest_y = quadBoundaryPoints(3,1);

  double width = southEast_x - southWest_x;
  double height = northWest_y - southWest_y;

  Teuchos::ParameterList pl;

  pl.set("useMinRule", false);
  pl.set("bf",bilinearForm);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("divideIntoTriangles", triangulate);
  pl.set("useConformingTraces", useConformingTraces);
  pl.set("trialOrderEnhancements", &trialOrderEnhancements);
  pl.set("testOrderEnhancements", &testOrderEnhancements);
  pl.set("x0",southWest_x);
  pl.set("y0",southWest_y);
  pl.set("width", width);
  pl.set("height",height);

  return quadMesh(pl);
}

MeshPtr MeshFactory::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
    int horizontalElements, int verticalElements,
    TBFPtr<double> bilinearForm,
    int H1Order, int pTest, bool useConformingTraces)
{
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order

  vector<vector<double> > vertices;
  vector< vector<IndexType> > allElementVertices;

  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                              std::invalid_argument,
                              "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");

  int numDimensions = 2;

  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1),
         southEast_x = quadBoundaryPoints(1,0),
         southEast_y = quadBoundaryPoints(1,1),
         northEast_x = quadBoundaryPoints(2,0),
         northEast_y = quadBoundaryPoints(2,1),
         northWest_x = quadBoundaryPoints(3,0),
         northWest_y = quadBoundaryPoints(3,1);

  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++)
  {
    for (int j=0; j<=verticalElements; j++)
    {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }

  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      bool triangulate = (i >= horizontalElements / 2); // triangles on right half of mesh
      if ( ! triangulate )
      {
        vector<IndexType> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
      else
      {
        vector<IndexType> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH

        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest,useConformingTraces));
}

void MeshFactory::quadMeshCellIDs(FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles)
{
  // populates cellIDs with either (h,v) or (h,v,2)
  // where h: horizontalElements (indexed by i, below)
  //       v: verticalElements   (indexed by j)
  //       2: triangles per quad (indexed by k)

  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(0)!=horizontalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(1)!=verticalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  if (useTriangles)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(2)!=2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 3,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements)");
  }

  int cellID = 0;
  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      if (useTriangles)
      {
        cellIDs(i,j,0) = cellID++;
        cellIDs(i,j,1) = cellID++;
      }
      else
      {
        cellIDs(i,j) = cellID++;
      }
    }
  }
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius)
{
  double meshHeight = yTop - yBottom;
  double embeddedSquareSideLength = cylinderRadius+meshHeight/2;
  return shiftedHemkerGeometry(xLeft, xRight, yBottom, yTop, cylinderRadius, embeddedSquareSideLength);
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius, double embeddedSquareSideLength)
{
  // first, set up an 8-element mesh, centered at the origin
  ParametricCurvePtr circle = ParametricCurve::circle(cylinderRadius, 0, 0);
  double meshHeight = yTop - yBottom;
  ParametricCurvePtr rect = parametricRect(embeddedSquareSideLength, embeddedSquareSideLength, 0, 0);

  int numPoints = 8; // 8 points on rect, 8 on circle
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++)
  {
    circle->value(t, innerVertices(i,0), innerVertices(i,1));
    rect  ->value(t, outerVertices(i,0), outerVertices(i,1));
    circle->value(t, innerVertex[0], innerVertex[1]);
    rect  ->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 1.0 / numPoints;
  }

  //  cout << "innerVertices:\n" << innerVertices;
  //  cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;
  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numPoints; i++)   // numPoints = numElements
  {
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

    //    cout << "innerIndex0: " << innerIndex0 << endl;
    //    cout << "innerIndex1: " << innerIndex1 << endl;
    //    cout << "outerIndex0: " << outerIndex0 << endl;
    //    cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(circle, t+1.0/numPoints, t);
    t += 1.0/numPoints;
  }

  int boundaryVertexOffset = vertices.size();
  // make some new vertices, going counter-clockwise:
  ParametricCurvePtr meshRect = parametricRect(xRight-xLeft, meshHeight, 0.5*(xLeft+xRight), 0.5*(yBottom + yTop));
  vector<double> boundaryVertex(spaceDim);
  boundaryVertex[0] = xRight;
  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = meshHeight / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -meshHeight / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = outerVertexIndices[0];
  vertexIndices[1] = boundaryVertexOffset;
  vertexIndices[2] = boundaryVertexOffset + 1;
  vertexIndices[3] = outerVertexIndices[1];
  elementVertices.push_back(vertexIndices);

  // mesh NE corner
  vertexIndices[0] = outerVertexIndices[1];
  vertexIndices[1] = boundaryVertexOffset + 1;
  vertexIndices[2] = boundaryVertexOffset + 2;
  vertexIndices[3] = boundaryVertexOffset + 3;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[2];
  vertexIndices[1] = outerVertexIndices[1];
  vertexIndices[2] = boundaryVertexOffset + 3;
  vertexIndices[3] = boundaryVertexOffset + 4;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[3];
  vertexIndices[1] = outerVertexIndices[2];
  vertexIndices[2] = boundaryVertexOffset + 4;
  vertexIndices[3] = boundaryVertexOffset + 5;
  elementVertices.push_back(vertexIndices);

  // NW corner
  vertexIndices[0] = boundaryVertexOffset + 7;
  vertexIndices[1] = outerVertexIndices[3];
  vertexIndices[2] = boundaryVertexOffset + 5;
  vertexIndices[3] = boundaryVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 8;
  vertexIndices[1] = outerVertexIndices[4];
  vertexIndices[2] = outerVertexIndices[3];
  vertexIndices[3] = boundaryVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 9;
  vertexIndices[1] = outerVertexIndices[5];
  vertexIndices[2] = outerVertexIndices[4];
  vertexIndices[3] = boundaryVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  // SW corner
  vertexIndices[0] = boundaryVertexOffset + 10;
  vertexIndices[1] = boundaryVertexOffset + 11;
  vertexIndices[2] = outerVertexIndices[5];
  vertexIndices[3] = boundaryVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 11;
  vertexIndices[1] = boundaryVertexOffset + 12;
  vertexIndices[2] = outerVertexIndices[6];
  vertexIndices[3] = outerVertexIndices[5];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 12;
  vertexIndices[1] = boundaryVertexOffset + 13;
  vertexIndices[2] = outerVertexIndices[7];
  vertexIndices[3] = outerVertexIndices[6];
  elementVertices.push_back(vertexIndices);

  // SE corner
  vertexIndices[0] = boundaryVertexOffset + 13;
  vertexIndices[1] = boundaryVertexOffset + 14;
  vertexIndices[2] = boundaryVertexOffset + 15;
  vertexIndices[3] = outerVertexIndices[7];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[7];
  vertexIndices[1] = boundaryVertexOffset + 15;
  vertexIndices[2] = boundaryVertexOffset;
  vertexIndices[3] = outerVertexIndices[0];
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshPtr MeshFactory::shiftedHemkerMesh(double xLeft, double xRight, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                       TBFPtr<double> bilinearForm, int H1Order, int pToAddTest)
{
  MeshGeometryPtr geometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, meshHeight, cylinderRadius);
  MeshPtr mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                        bilinearForm, H1Order, pToAddTest) );

  map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = geometry->edgeToCurveMap();
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
  mesh->setEdgeToCurveMap(globalEdgeToCurveMap);
  return mesh;
}

MeshPtr MeshFactory::spaceTimeMesh(MeshTopologyPtr spatialMeshTopology, double t0, double t1,
                                   TBFPtr<double> bf, int spatialH1Order, int temporalH1Order, int pToAdd,
                                   Epetra_CommPtr Comm)
{
  MeshTopologyPtr meshTopology = spaceTimeMeshTopology(spatialMeshTopology, t0, t1); // for refined spatial topologies, this can be more than 1-irregular--we enforce 1-irregularity below.

  vector<int> H1Order(2);
  H1Order[0] = spatialH1Order;
  H1Order[1] = temporalH1Order;

  map<int,int> emptyMap;
  MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd, emptyMap, emptyMap, nullPartitionPolicy, Comm) );
  mesh->enforceOneIrregularity();
  
  return mesh;
}

MeshGeometryPtr MeshFactory::halfHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius)
{
  // first, set up an 4-element mesh, centered at the origin
  ParametricCurvePtr circle = ParametricCurve::circle(cylinderRadius, 0, 0);
  ParametricCurvePtr rect = parametricRect(2.0*meshHeight, 2.0*meshHeight, 0, 0);  
  // ParametricCurvePtr semiCircle = ParametricCurve::subCurve(circle, 0, 0.5);
  // ParametricCurvePtr semiRect = ParametricCurve::subCurve(rect, 0, 0.5); // does not seem to work with subCurve


  //  __ _______________ __ 
  //    |`      :      '|    |
  //    |  ` .__:__. '  |    |- mesh height
  //    |    /     \    |    |
  //  __|___/       \___|__  |
  //

  int numPoints = 5; // 5 points on rectangle, 5 on circle
  int numElements = numPoints - 1;
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++)
  {
    circle->value(t, innerVertices(i,0), innerVertices(i,1));
    rect  ->value(t, outerVertices(i,0), outerVertices(i,1));
    circle->value(t, innerVertex[0], innerVertex[1]);
    rect  ->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 0.5 / numElements;
  }

   // cout << "innerVertices:\n" << innerVertices;
   // cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;

  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numElements; i++)
  {
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

       // cout << "innerIndex0: " << innerIndex0 << endl;
       // cout << "innerIndex1: " << innerIndex1 << endl;
       // cout << "outerIndex0: " << outerIndex0 << endl;
       // cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(circle, t+0.5/numElements, t);
    t += 0.5 / numElements;
  }

  //  Full mesh:
  //   _ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ _
  //  | |   |   |   |   |   |   |\_/|   |   |   |   |   |   | |
  //  |_|___|___|___|___|___|___|| ||___|___|___|___|___|___|_|
  //

  int boundaryVertexOffset = vertices.size();
  // make some new vertices, going counter-clockwise:

  // ParametricCurvePtr meshRect = parametricRect(xRight-xLeft, meshHeight, 0.5*(xLeft+xRight), 0); // is this necessary?

  vector<double> boundaryVertex(spaceDim);
  boundaryVertex[0] = xRight;
  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = meshHeight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (14.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (12.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (10.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (8.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (6.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (4.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = (2.0/15.0)*xRight;
  // vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = 0.0;
  // vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = (2.0/15.0)*xLeft;
  // vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (4.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (6.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (8.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (10.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (12.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (14.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (14.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (12.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (10.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (8.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (6.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (4.0/15.0)*xLeft;
  vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = (2.0/15.0)*xLeft;
  // vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = 0.0;
  // vertices.push_back(boundaryVertex);

  // boundaryVertex[0] = (2.0/15.0)*xRight;
  // vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (4.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (6.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (8.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (10.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (12.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = (14.0/15.0)*xRight;
  vertices.push_back(boundaryVertex);

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = boundaryVertexOffset;
  vertexIndices[1] = boundaryVertexOffset + 1;
  vertexIndices[2] = boundaryVertexOffset + 2;
  vertexIndices[3] = boundaryVertexOffset + 27;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 2;
  vertexIndices[1] = boundaryVertexOffset + 3;
  vertexIndices[2] = boundaryVertexOffset + 26;
  vertexIndices[3] = boundaryVertexOffset + 27;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 3;
  vertexIndices[1] = boundaryVertexOffset + 4;
  vertexIndices[2] = boundaryVertexOffset + 25;
  vertexIndices[3] = boundaryVertexOffset + 26;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 4;
  vertexIndices[1] = boundaryVertexOffset + 5;
  vertexIndices[2] = boundaryVertexOffset + 24;
  vertexIndices[3] = boundaryVertexOffset + 25;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 5;
  vertexIndices[1] = boundaryVertexOffset + 6;
  vertexIndices[2] = boundaryVertexOffset + 23;
  vertexIndices[3] = boundaryVertexOffset + 24;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 6;
  vertexIndices[1] = boundaryVertexOffset + 7;
  vertexIndices[2] = boundaryVertexOffset + 22;
  vertexIndices[3] = boundaryVertexOffset + 23;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 7;
  vertexIndices[1] = outerVertexIndices[1];
  vertexIndices[2] = outerVertexIndices[0];
  vertexIndices[3] = boundaryVertexOffset + 22;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[3];
  vertexIndices[1] = boundaryVertexOffset + 8;
  vertexIndices[2] = boundaryVertexOffset + 21;
  vertexIndices[3] = outerVertexIndices[4];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 8;
  vertexIndices[1] = boundaryVertexOffset + 9;
  vertexIndices[2] = boundaryVertexOffset + 20;
  vertexIndices[3] = boundaryVertexOffset + 21;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 9;
  vertexIndices[1] = boundaryVertexOffset + 10;
  vertexIndices[2] = boundaryVertexOffset + 19;
  vertexIndices[3] = boundaryVertexOffset + 20;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 10;
  vertexIndices[1] = boundaryVertexOffset + 11;
  vertexIndices[2] = boundaryVertexOffset + 18;
  vertexIndices[3] = boundaryVertexOffset + 19;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 11;
  vertexIndices[1] = boundaryVertexOffset + 12;
  vertexIndices[2] = boundaryVertexOffset + 17;
  vertexIndices[3] = boundaryVertexOffset + 18;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 12;
  vertexIndices[1] = boundaryVertexOffset + 13;
  vertexIndices[2] = boundaryVertexOffset + 16;
  vertexIndices[3] = boundaryVertexOffset + 17;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 13;
  vertexIndices[1] = boundaryVertexOffset + 14;
  vertexIndices[2] = boundaryVertexOffset + 15;
  vertexIndices[3] = boundaryVertexOffset + 16;
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshGeometryPtr MeshFactory::halfConfinedCylinderGeometry(double cylinderRadius)
{
  double yMax = 2.0*cylinderRadius;

  // first, set up an 5-element mesh, centered at the origin
  ParametricCurvePtr innerCircle = ParametricCurve::circle(cylinderRadius, 0, 0);
  double rOuter = sqrt(2 + 2 / sqrt(5))*cylinderRadius;
  ParametricCurvePtr outerCircle = ParametricCurve::circle(rOuter, 0, 0);

  int numPoints = 6; // 6 on circles
  int numElements = numPoints - 1;
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++)
  {
    innerCircle->value(t, innerVertices(i,0), innerVertices(i,1));
    outerCircle->value(t, outerVertices(i,0), outerVertices(i,1));
    innerCircle->value(t, innerVertex[0], innerVertex[1]);
    outerCircle->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 0.5 / numElements;
  }

   // cout << "innerVertices:\n" << innerVertices;
   // cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;

  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numElements; i++)
  {
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

       // cout << "innerIndex0: " << innerIndex0 << endl;
       // cout << "innerIndex1: " << innerIndex1 << endl;
       // cout << "outerIndex0: " << outerIndex0 << endl;
       // cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(innerCircle, t+0.5/numElements, t);
    t += 0.5 / numElements;
  }

  // make some new vertices, going counter-clockwise in the mesh near the cylinder

  vector<double> nearCylinderVertex(spaceDim);
  double rCosTheta = sqrt(1 + 2 / sqrt(5));
  double rCosTwoTheta = sqrt((5 - sqrt(5))/10);

  nearCylinderVertex[0] = 3.0*cylinderRadius;
  nearCylinderVertex[1] = 0;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-3.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = 0.0;
  vertices.push_back(nearCylinderVertex);

  // add the new elements to the mesh

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = 12;
  vertexIndices[1] = 13;
  vertexIndices[2] = 3;
  vertexIndices[3] = 1;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 13;
  vertexIndices[1] = 14;
  vertexIndices[2] = 15;
  vertexIndices[3] = 3;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 3;
  vertexIndices[1] = 15;
  vertexIndices[2] = 16;
  vertexIndices[3] = 5;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 5;
  vertexIndices[1] = 16;
  vertexIndices[2] = 17;
  vertexIndices[3] = 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 7;
  vertexIndices[1] = 17;
  vertexIndices[2] = 18;
  vertexIndices[3] = 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 9;
  vertexIndices[1] = 18;
  vertexIndices[2] = 19;
  vertexIndices[3] = 20;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 11;
  vertexIndices[1] = 9;
  vertexIndices[2] = 20;
  vertexIndices[3] = 21;
  elementVertices.push_back(vertexIndices);

  int nearCylinderVertexOffset = vertices.size();

  vector<double> inflowVertex(spaceDim);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = 0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  // add inflow elements

  vertexIndices[0] = nearCylinderVertexOffset + 7;
  vertexIndices[1] = nearCylinderVertexOffset + 1;
  vertexIndices[2] = nearCylinderVertexOffset;
  vertexIndices[3] = nearCylinderVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 8;
  vertexIndices[1] = nearCylinderVertexOffset + 2;
  vertexIndices[2] = nearCylinderVertexOffset + 1;
  vertexIndices[3] = nearCylinderVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 9;
  vertexIndices[1] = nearCylinderVertexOffset + 3;
  vertexIndices[2] = nearCylinderVertexOffset + 2;
  vertexIndices[3] = nearCylinderVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 10;
  vertexIndices[1] = nearCylinderVertexOffset + 4;
  vertexIndices[2] = nearCylinderVertexOffset + 3;
  vertexIndices[3] = nearCylinderVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 11;
  vertexIndices[1] = nearCylinderVertexOffset + 5;
  vertexIndices[2] = nearCylinderVertexOffset + 4;
  vertexIndices[3] = nearCylinderVertexOffset + 10;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 20;
  vertexIndices[1] = 19;
  vertexIndices[2] = nearCylinderVertexOffset + 5;
  vertexIndices[3] = nearCylinderVertexOffset + 11;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 13;
  vertexIndices[1] = nearCylinderVertexOffset + 7;
  vertexIndices[2] = nearCylinderVertexOffset + 6;
  vertexIndices[3] = nearCylinderVertexOffset + 12;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 14;
  vertexIndices[1] = nearCylinderVertexOffset + 8;
  vertexIndices[2] = nearCylinderVertexOffset + 7;
  vertexIndices[3] = nearCylinderVertexOffset + 13;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 15;
  vertexIndices[1] = nearCylinderVertexOffset + 9;
  vertexIndices[2] = nearCylinderVertexOffset + 8;
  vertexIndices[3] = nearCylinderVertexOffset + 14;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 16;
  vertexIndices[1] = nearCylinderVertexOffset + 10;
  vertexIndices[2] = nearCylinderVertexOffset + 9;
  vertexIndices[3] = nearCylinderVertexOffset + 15;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 17;
  vertexIndices[1] = nearCylinderVertexOffset + 11;
  vertexIndices[2] = nearCylinderVertexOffset + 10;
  vertexIndices[3] = nearCylinderVertexOffset + 16;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 21;
  vertexIndices[1] = 20;
  vertexIndices[2] = nearCylinderVertexOffset + 11;
  vertexIndices[3] = nearCylinderVertexOffset + 17;
  elementVertices.push_back(vertexIndices);

  int inflowVertexOffset = vertices.size();

  vector<double> outflowVertex(spaceDim);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = 0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  // add inflow elements

  vertexIndices[0] = inflowVertexOffset + 6;
  vertexIndices[1] = inflowVertexOffset;
  vertexIndices[2] = 14;
  vertexIndices[3] = 13;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 7;
  vertexIndices[1] = inflowVertexOffset + 1;
  vertexIndices[2] = inflowVertexOffset;
  vertexIndices[3] = inflowVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 8;
  vertexIndices[1] = inflowVertexOffset + 2;
  vertexIndices[2] = inflowVertexOffset + 1;
  vertexIndices[3] = inflowVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 9;
  vertexIndices[1] = inflowVertexOffset + 3;
  vertexIndices[2] = inflowVertexOffset + 2;
  vertexIndices[3] = inflowVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 10;
  vertexIndices[1] = inflowVertexOffset + 4;
  vertexIndices[2] = inflowVertexOffset + 3;
  vertexIndices[3] = inflowVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 11;
  vertexIndices[1] = inflowVertexOffset + 5;
  vertexIndices[2] = inflowVertexOffset + 4;
  vertexIndices[3] = inflowVertexOffset + 10;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 12;
  vertexIndices[1] = inflowVertexOffset + 6;
  vertexIndices[2] = 13;
  vertexIndices[3] = 12;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 13;
  vertexIndices[1] = inflowVertexOffset + 7;
  vertexIndices[2] = inflowVertexOffset + 6;
  vertexIndices[3] = inflowVertexOffset + 12;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 14;
  vertexIndices[1] = inflowVertexOffset + 8;
  vertexIndices[2] = inflowVertexOffset + 7;
  vertexIndices[3] = inflowVertexOffset + 13;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 15;
  vertexIndices[1] = inflowVertexOffset + 9;
  vertexIndices[2] = inflowVertexOffset + 8;
  vertexIndices[3] = inflowVertexOffset + 14;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 16;
  vertexIndices[1] = inflowVertexOffset + 10;
  vertexIndices[2] = inflowVertexOffset + 9;
  vertexIndices[3] = inflowVertexOffset + 15;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 17;
  vertexIndices[1] = inflowVertexOffset + 11;
  vertexIndices[2] = inflowVertexOffset + 10;
  vertexIndices[3] = inflowVertexOffset + 16;
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshGeometryPtr MeshFactory::confinedCylinderGeometry(double cylinderRadius)
{
  double yMax = 2.0*cylinderRadius;

  // first, set up an 10-element mesh, centered at the origin
  ParametricCurvePtr innerCircle = ParametricCurve::circle(cylinderRadius, 0, 0);
  double rOuter = sqrt(2 + 2 / sqrt(5))*cylinderRadius;
  ParametricCurvePtr outerCircle = ParametricCurve::circle(rOuter, 0, 0);

  int numPoints = 10; // 10 on circles
  int numElements = numPoints;
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++)
  {
    innerCircle->value(t, innerVertices(i,0), innerVertices(i,1));
    outerCircle->value(t, outerVertices(i,0), outerVertices(i,1));
    innerCircle->value(t, innerVertex[0], innerVertex[1]);
    outerCircle->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 1.0 / numElements;
  }

   // cout << "innerVertices:\n" << innerVertices;
   // cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;

  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numElements; i++)
  {
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

       // cout << "innerIndex0: " << innerIndex0 << endl;
       // cout << "innerIndex1: " << innerIndex1 << endl;
       // cout << "outerIndex0: " << outerIndex0 << endl;
       // cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(innerCircle, t+1.0/numElements, t);
    t += 1.0 / numElements;
  }

  // int nearCylinderVertexOffset = vertices.size();
  // make some new vertices, going counter-clockwise in the mesh near the cylinder

  vector<double> nearCylinderVertex(spaceDim);
  double rCosTheta = sqrt(1 + 2 / sqrt(5));
  double rCosTwoTheta = sqrt((5 - sqrt(5))/10);

  nearCylinderVertex[0] = 3.0*cylinderRadius;
  nearCylinderVertex[1] = 0;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-3.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] = 0.0;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] =-cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] =-2.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] =-rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTwoTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = rCosTheta;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[0] = 3.0*cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  nearCylinderVertex[1] =-cylinderRadius;
  vertices.push_back(nearCylinderVertex);

  // add the new elements to the mesh

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = 20;
  vertexIndices[1] = 21;
  vertexIndices[2] = 3;
  vertexIndices[3] = 1;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 21;
  vertexIndices[1] = 22;
  vertexIndices[2] = 23;
  vertexIndices[3] = 3;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 3;
  vertexIndices[1] = 23;
  vertexIndices[2] = 24;
  vertexIndices[3] = 5;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 5;
  vertexIndices[1] = 24;
  vertexIndices[2] = 25;
  vertexIndices[3] = 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 7;
  vertexIndices[1] = 25;
  vertexIndices[2] = 26;
  vertexIndices[3] = 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 9;
  vertexIndices[1] = 26;
  vertexIndices[2] = 27;
  vertexIndices[3] = 28;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 11;
  vertexIndices[1] = 9;
  vertexIndices[2] = 28;
  vertexIndices[3] = 29;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 13;
  vertexIndices[1] = 11;
  vertexIndices[2] = 29;
  vertexIndices[3] = 30;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 32;
  vertexIndices[1] = 13;
  vertexIndices[2] = 30;
  vertexIndices[3] = 31;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 33;
  vertexIndices[1] = 15;
  vertexIndices[2] = 13;
  vertexIndices[3] = 32;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 34;
  vertexIndices[1] = 17;
  vertexIndices[2] = 15;
  vertexIndices[3] = 33;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 35;
  vertexIndices[1] = 19;
  vertexIndices[2] = 17;
  vertexIndices[3] = 34;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 36;
  vertexIndices[1] = 37;
  vertexIndices[2] = 19;
  vertexIndices[3] = 35;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 37;
  vertexIndices[1] = 20;
  vertexIndices[2] = 1;
  vertexIndices[3] = 19;
  elementVertices.push_back(vertexIndices);

  int nearCylinderVertexOffset = vertices.size();

  vector<double> inflowVertex(spaceDim);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] = 0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] =-cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-15.0;
  inflowVertex[1] =-2.0*cylinderRadius;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-13.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-11.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-9.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-7.0;
  vertices.push_back(inflowVertex);

  inflowVertex[0] =-5.0;
  vertices.push_back(inflowVertex);

  // add inflow elements

  vertexIndices[0] = nearCylinderVertexOffset + 7;
  vertexIndices[1] = nearCylinderVertexOffset + 1;
  vertexIndices[2] = nearCylinderVertexOffset;
  vertexIndices[3] = nearCylinderVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 8;
  vertexIndices[1] = nearCylinderVertexOffset + 2;
  vertexIndices[2] = nearCylinderVertexOffset + 1;
  vertexIndices[3] = nearCylinderVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 9;
  vertexIndices[1] = nearCylinderVertexOffset + 3;
  vertexIndices[2] = nearCylinderVertexOffset + 2;
  vertexIndices[3] = nearCylinderVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 10;
  vertexIndices[1] = nearCylinderVertexOffset + 4;
  vertexIndices[2] = nearCylinderVertexOffset + 3;
  vertexIndices[3] = nearCylinderVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 11;
  vertexIndices[1] = nearCylinderVertexOffset + 5;
  vertexIndices[2] = nearCylinderVertexOffset + 4;
  vertexIndices[3] = nearCylinderVertexOffset + 10;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 28;
  vertexIndices[1] = 27;
  vertexIndices[2] = nearCylinderVertexOffset + 5;
  vertexIndices[3] = nearCylinderVertexOffset + 11;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 13;
  vertexIndices[1] = nearCylinderVertexOffset + 7;
  vertexIndices[2] = nearCylinderVertexOffset + 6;
  vertexIndices[3] = nearCylinderVertexOffset + 12;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 14;
  vertexIndices[1] = nearCylinderVertexOffset + 8;
  vertexIndices[2] = nearCylinderVertexOffset + 7;
  vertexIndices[3] = nearCylinderVertexOffset + 13;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 15;
  vertexIndices[1] = nearCylinderVertexOffset + 9;
  vertexIndices[2] = nearCylinderVertexOffset + 8;
  vertexIndices[3] = nearCylinderVertexOffset + 14;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 16;
  vertexIndices[1] = nearCylinderVertexOffset + 10;
  vertexIndices[2] = nearCylinderVertexOffset + 9;
  vertexIndices[3] = nearCylinderVertexOffset + 15;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 17;
  vertexIndices[1] = nearCylinderVertexOffset + 11;
  vertexIndices[2] = nearCylinderVertexOffset + 10;
  vertexIndices[3] = nearCylinderVertexOffset + 16;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 29;
  vertexIndices[1] = 28;
  vertexIndices[2] = nearCylinderVertexOffset + 11;
  vertexIndices[3] = nearCylinderVertexOffset + 17;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 19;
  vertexIndices[1] = nearCylinderVertexOffset + 13;
  vertexIndices[2] = nearCylinderVertexOffset + 12;
  vertexIndices[3] = nearCylinderVertexOffset + 18;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 20;
  vertexIndices[1] = nearCylinderVertexOffset + 14;
  vertexIndices[2] = nearCylinderVertexOffset + 13;
  vertexIndices[3] = nearCylinderVertexOffset + 19;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 21;
  vertexIndices[1] = nearCylinderVertexOffset + 15;
  vertexIndices[2] = nearCylinderVertexOffset + 14;
  vertexIndices[3] = nearCylinderVertexOffset + 20;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 22;
  vertexIndices[1] = nearCylinderVertexOffset + 16;
  vertexIndices[2] = nearCylinderVertexOffset + 15;
  vertexIndices[3] = nearCylinderVertexOffset + 21;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 23;
  vertexIndices[1] = nearCylinderVertexOffset + 17;
  vertexIndices[2] = nearCylinderVertexOffset + 16;
  vertexIndices[3] = nearCylinderVertexOffset + 22;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 30;
  vertexIndices[1] = 29;
  vertexIndices[2] = nearCylinderVertexOffset + 17;
  vertexIndices[3] = nearCylinderVertexOffset + 23;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 25;
  vertexIndices[1] = nearCylinderVertexOffset + 19;
  vertexIndices[2] = nearCylinderVertexOffset + 18;
  vertexIndices[3] = nearCylinderVertexOffset + 24;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 26;
  vertexIndices[1] = nearCylinderVertexOffset + 20;
  vertexIndices[2] = nearCylinderVertexOffset + 19;
  vertexIndices[3] = nearCylinderVertexOffset + 25;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 27;
  vertexIndices[1] = nearCylinderVertexOffset + 21;
  vertexIndices[2] = nearCylinderVertexOffset + 20;
  vertexIndices[3] = nearCylinderVertexOffset + 26;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 28;
  vertexIndices[1] = nearCylinderVertexOffset + 22;
  vertexIndices[2] = nearCylinderVertexOffset + 21;
  vertexIndices[3] = nearCylinderVertexOffset + 27;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = nearCylinderVertexOffset + 29;
  vertexIndices[1] = nearCylinderVertexOffset + 23;
  vertexIndices[2] = nearCylinderVertexOffset + 22;
  vertexIndices[3] = nearCylinderVertexOffset + 28;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = 31;
  vertexIndices[1] = 30;
  vertexIndices[2] = nearCylinderVertexOffset + 23;
  vertexIndices[3] = nearCylinderVertexOffset + 29;
  elementVertices.push_back(vertexIndices);

  int inflowVertexOffset = vertices.size();

  vector<double> outflowVertex(spaceDim);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = 2.0*cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] = 0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] =-cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 5.0;
  outflowVertex[1] =-2.0*cylinderRadius;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 7.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 9.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 11.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 13.0;
  vertices.push_back(outflowVertex);

  outflowVertex[0] = 15.0;
  vertices.push_back(outflowVertex);


  // add inflow elements

  vertexIndices[0] = inflowVertexOffset + 6;
  vertexIndices[1] = inflowVertexOffset;
  vertexIndices[2] = 22;
  vertexIndices[3] = 21;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 7;
  vertexIndices[1] = inflowVertexOffset + 1;
  vertexIndices[2] = inflowVertexOffset;
  vertexIndices[3] = inflowVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 8;
  vertexIndices[1] = inflowVertexOffset + 2;
  vertexIndices[2] = inflowVertexOffset + 1;
  vertexIndices[3] = inflowVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 9;
  vertexIndices[1] = inflowVertexOffset + 3;
  vertexIndices[2] = inflowVertexOffset + 2;
  vertexIndices[3] = inflowVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 10;
  vertexIndices[1] = inflowVertexOffset + 4;
  vertexIndices[2] = inflowVertexOffset + 3;
  vertexIndices[3] = inflowVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 11;
  vertexIndices[1] = inflowVertexOffset + 5;
  vertexIndices[2] = inflowVertexOffset + 4;
  vertexIndices[3] = inflowVertexOffset + 10;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 12;
  vertexIndices[1] = inflowVertexOffset + 6;
  vertexIndices[2] = 21;
  vertexIndices[3] = 20;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 13;
  vertexIndices[1] = inflowVertexOffset + 7;
  vertexIndices[2] = inflowVertexOffset + 6;
  vertexIndices[3] = inflowVertexOffset + 12;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 14;
  vertexIndices[1] = inflowVertexOffset + 8;
  vertexIndices[2] = inflowVertexOffset + 7;
  vertexIndices[3] = inflowVertexOffset + 13;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 15;
  vertexIndices[1] = inflowVertexOffset + 9;
  vertexIndices[2] = inflowVertexOffset + 8;
  vertexIndices[3] = inflowVertexOffset + 14;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 16;
  vertexIndices[1] = inflowVertexOffset + 10;
  vertexIndices[2] = inflowVertexOffset + 9;
  vertexIndices[3] = inflowVertexOffset + 15;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 17;
  vertexIndices[1] = inflowVertexOffset + 11;
  vertexIndices[2] = inflowVertexOffset + 10;
  vertexIndices[3] = inflowVertexOffset + 16;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 18;
  vertexIndices[1] = inflowVertexOffset + 12;
  vertexIndices[2] = 20;
  vertexIndices[3] = 37;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 19;
  vertexIndices[1] = inflowVertexOffset + 13;
  vertexIndices[2] = inflowVertexOffset + 12;
  vertexIndices[3] = inflowVertexOffset + 18;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 20;
  vertexIndices[1] = inflowVertexOffset + 14;
  vertexIndices[2] = inflowVertexOffset + 13;
  vertexIndices[3] = inflowVertexOffset + 19;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 21;
  vertexIndices[1] = inflowVertexOffset + 15;
  vertexIndices[2] = inflowVertexOffset + 14;
  vertexIndices[3] = inflowVertexOffset + 20;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 22;
  vertexIndices[1] = inflowVertexOffset + 16;
  vertexIndices[2] = inflowVertexOffset + 15;
  vertexIndices[3] = inflowVertexOffset + 21;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 23;
  vertexIndices[1] = inflowVertexOffset + 17;
  vertexIndices[2] = inflowVertexOffset + 16;
  vertexIndices[3] = inflowVertexOffset + 22;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 24;
  vertexIndices[1] = inflowVertexOffset + 18;
  vertexIndices[2] = 37;
  vertexIndices[3] = 36;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 25;
  vertexIndices[1] = inflowVertexOffset + 19;
  vertexIndices[2] = inflowVertexOffset + 18;
  vertexIndices[3] = inflowVertexOffset + 24;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 26;
  vertexIndices[1] = inflowVertexOffset + 20;
  vertexIndices[2] = inflowVertexOffset + 19;
  vertexIndices[3] = inflowVertexOffset + 25;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 27;
  vertexIndices[1] = inflowVertexOffset + 21;
  vertexIndices[2] = inflowVertexOffset + 20;
  vertexIndices[3] = inflowVertexOffset + 26;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 28;
  vertexIndices[1] = inflowVertexOffset + 22;
  vertexIndices[2] = inflowVertexOffset + 21;
  vertexIndices[3] = inflowVertexOffset + 27;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = inflowVertexOffset + 29;
  vertexIndices[1] = inflowVertexOffset + 23;
  vertexIndices[2] = inflowVertexOffset + 22;
  vertexIndices[3] = inflowVertexOffset + 28;
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshTopologyPtr MeshFactory::spaceTimeMeshTopology(MeshTopologyPtr spatialMeshTopology, double t0, double t1, int temporalDivisions)
{
  // we allow spatialMeshTopology to have been refined; we start with a coarse space-time topology matching the root spatial topology,
  // and then refine accordingly...

  // (For now, though, we do make the assumption that all refinements are regular (isotropic).)

  int spaceDim = spatialMeshTopology->getDimension();
  int spaceTimeDim = spaceDim + 1;

  set<IndexType> rootCellIndices = spatialMeshTopology->getRootCellIndicesGlobal();
  
  MeshTopologyViewPtr rootSpatialTopology = spatialMeshTopology->getView(rootCellIndices);
  MeshTopologyPtr spaceTimeTopology = Teuchos::rcp( new MeshTopology( spaceTimeDim ));

  /*
   This is something of a conceit, but it's nice if the vertex indices in the space-time mesh topology are
   in the following relationship to the spatialMeshTopology:

   If v is a vertexIndex in spatialMeshTopology and spatialMeshTopology has N vertices, then
   - (v,t0) has vertexIndex v in spaceTimeMeshTopology, and
   - (v,t1) has vertexIndex v+N in spaceTimeMeshTopology.
  */

  IndexType N = spatialMeshTopology->getEntityCount(0);
  for (int timeSubdivision=0; timeSubdivision<temporalDivisions; timeSubdivision++)
  {
    vector<double> spaceTimeVertex(spaceTimeDim);
    FieldContainer<double> timeValues(2,1);
    timeValues[0] = t0 + timeSubdivision * (t1-t0) / temporalDivisions;
    timeValues[1] = t0 + (timeSubdivision+1) * (t1-t0) / temporalDivisions;
    for (int i=0; i<timeValues.size(); i++)
    {
      for (IndexType vertexIndex=0; vertexIndex<N; vertexIndex++)
      {
        const vector<double> *spaceVertex = &spatialMeshTopology->getVertex(vertexIndex);
        for (int d=0; d<spaceDim; d++)
        {
          spaceTimeVertex[d] = (*spaceVertex)[d];
        }
        spaceTimeVertex[spaceDim] = timeValues(i,0);
        spaceTimeTopology->addVertex(spaceTimeVertex);
      }
    }
  }

  // for now, we only do refinements on the first temporal subdivision
  // later, we might want to enforce 1-irregularity, at least
  set<IndexType> cellIndices = rootSpatialTopology->baseMeshTopology()->getRootCellIndicesGlobal();
  int tensorialDegree = 1;
  vector< FieldContainer<double> > componentNodes(2);
  FieldContainer<double> spatialCellNodes;
  FieldContainer<double> spaceTimeCellNodes;

  map<IndexType,IndexType> cellIDMap; // from space-time ID (in first temporal subdivision) to corresponding spatial ID

  for (int timeSubdivision=0; timeSubdivision<temporalDivisions; timeSubdivision++)
  {
    FieldContainer<double> timeValues(2,1);
    timeValues[0] = t0 + timeSubdivision * (t1-t0) / temporalDivisions;
    timeValues[1] = t0 + (timeSubdivision+1) * (t1-t0) / temporalDivisions;
    componentNodes[1] = timeValues;

    for (set<IndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++)
    {
      IndexType cellIndex = *cellIt;
      CellPtr spatialCell = rootSpatialTopology->getCell(cellIndex);
      CellTopoPtr spaceTimeCellTopology = CellTopology::cellTopology(spatialCell->topology(), tensorialDegree);
      int vertexCount = spatialCell->topology()->getVertexCount();

      spatialCellNodes.resize(vertexCount,spaceDim);
      const vector<IndexType>* vertexIndices = &spatialCell->vertices();
      for (int vertex=0; vertex<vertexCount; vertex++)
      {
        IndexType vertexIndex = (*vertexIndices)[vertex];
        for (int i=0; i<spaceDim; i++)
        {
          spatialCellNodes(vertex,i) = spatialMeshTopology->getVertex(vertexIndex)[i];
        }
      }
      componentNodes[0] = spatialCellNodes;

      spaceTimeCellNodes.resize(spaceTimeCellTopology->getVertexCount(),spaceTimeDim);
      spaceTimeCellTopology->initializeNodes(componentNodes, spaceTimeCellNodes);

      CellPtr spaceTimeCell = spaceTimeTopology->addCell(spaceTimeCellTopology, spaceTimeCellNodes);
      if (timeSubdivision==0) cellIDMap[spaceTimeCell->cellIndex()] = cellIndex;
    }
  }
  
  // construct the initial time entity set:
  // cellIDMap keys are the space-time cell IDs that match the initial time
  // we want the sides that correspond to the initial time
  EntitySetPtr initialTimeSet = spaceTimeTopology->createEntitySet();
  for (auto entry : cellIDMap)
  {
    IndexType initialTimeCellID = entry.first;
    CellPtr spaceTimeCell = spaceTimeTopology->getCell(initialTimeCellID);
    unsigned sideOrdinal = spaceTimeCell->topology()->getTemporalSideOrdinal(0);
    IndexType sideEntityIndex = spaceTimeCell->entityIndex(spaceDim, sideOrdinal);
    initialTimeSet->addEntity(spaceDim, sideEntityIndex);
  }
  spaceTimeTopology->setEntitySetInitialTime(initialTimeSet);

  bool noCellsToRefine = false;

  GlobalIndexType newCellID = spaceTimeTopology->cellCount();
  
  while (!noCellsToRefine)
  {
    noCellsToRefine = true;

    // TODO: check whether the set conversion here is necessary; putting it right now to avoid changing the effect of the code below.  (The vector is not sorted by cell index, but by owning MPI rank followed by cell index.  So there certainly could be some change in the execution.)
    vector<IndexType> activeSpaceTimeCellIndicesVector = spaceTimeTopology->getActiveCellIndicesGlobal();
    set<IndexType> activeSpaceTimeCellIndices(activeSpaceTimeCellIndicesVector.begin(), activeSpaceTimeCellIndicesVector.end());
    for (set<IndexType>::iterator cellIt = activeSpaceTimeCellIndices.begin(); cellIt != activeSpaceTimeCellIndices.end(); cellIt++)
    {
      IndexType spaceTimeCellIndex = *cellIt;
      if (cellIDMap.find(spaceTimeCellIndex) != cellIDMap.end())
      {
        IndexType spatialCellIndex = cellIDMap[spaceTimeCellIndex];
        CellPtr spatialCell = spatialMeshTopology->getCell(spatialCellIndex);
        if (spatialCell->isParent(spatialMeshTopology))
        {
          noCellsToRefine = false; // indicate we refined some on this pass...

          CellPtr spaceTimeCell = spaceTimeTopology->getCell(*cellIt);
          RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(spaceTimeCell->topology());
          spaceTimeTopology->refineCell(spaceTimeCellIndex, refPattern, newCellID);
          
          newCellID += refPattern->numChildren();

          vector<CellPtr> spatialChildren = spatialCell->children();
          for (int childOrdinal=0; childOrdinal<spatialChildren.size(); childOrdinal++)
          {
            CellPtr spatialChild = spatialChildren[childOrdinal];
            int vertexCount = spatialChild->topology()->getVertexCount();

            vector< vector<double> > childNodes(vertexCount);

            spatialCellNodes.resize(vertexCount,spaceDim);
            const vector<IndexType>* vertexIndices = &spatialChild->vertices();
            for (int vertex=0; vertex<vertexCount; vertex++)
            {
              IndexType vertexIndex = (*vertexIndices)[vertex];
              childNodes[vertex] = spatialMeshTopology->getVertex(vertexIndex);
              childNodes[vertex].push_back(t0);
            }

            CellPtr spaceTimeChild = spaceTimeTopology->findCellWithVertices(childNodes);
            cellIDMap[spaceTimeChild->cellIndex()] = spatialChild->cellIndex();
          }
        }
      }
    }
  }

  return spaceTimeTopology;
}
