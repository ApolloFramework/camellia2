//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ParametricSurface.h"
#include "BasisFactory.h"
#include "IP.h"
#include "VarFactory.h"
#include "Projector.h"
#include "BasisSumFunction.h"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

using namespace Intrepid;
using namespace Camellia;

class LinearInterpolatingSurface : public ParametricSurface
{
  vector< pair<double, double> > _vertices;
public:
  LinearInterpolatingSurface(const vector< ParametricCurvePtr > &curves)
  {
    for (int i=0; i<curves.size(); i++)
    {
      _vertices.push_back(make_pair(0,0));
      curves[i]->value(0, _vertices[i].first, _vertices[i].second);
    }
  }
  std::vector< TFunctionPtr<double> > memberFunctions()
  {
    return std::vector< TFunctionPtr<double> >();
  }
  
  void value(double t1, double t2, double &x, double &y);
};

void LinearInterpolatingSurface::value(double t1, double t2, double &x, double &y)
{
  if (_vertices.size() == 4)
  {
    x  = _vertices[0].first*(1-t1)*(1-t2) + _vertices[1].first*   t1 *(1-t2)
         + _vertices[2].first*   t1*    t2  + _vertices[3].first*(1-t1)*   t2;
    y  = _vertices[0].second*(1-t1)*(1-t2) + _vertices[1].second*   t1 *(1-t2)
         + _vertices[2].second*   t1*    t2  + _vertices[3].second*(1-t1)*   t2;
  }
  else if (_vertices.size() == 3)
  {
    // TODO: implement this
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads and (eventually) triangles supported...");
  }
}

class TransfiniteInterpolatingSurface : public ParametricSurface
{
  vector< ParametricCurvePtr > _curves;
  vector< pair<double, double> > _vertices;
  bool _neglectVertices; // if true, then the value returned by value() is a "bubble" value...
  Camellia::EOperator _op;

  void init(const vector< ParametricCurvePtr > &curves, Camellia::EOperator op,
            const vector< pair<double, double> > &vertices)
  {
    if ((op != OP_VALUE) && (op != OP_DX) && (op != OP_DY))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported operator");
    }
    if (curves.size() != 4)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
    }

    _neglectVertices = false;
    _curves = curves;
    _op = op;

    if (vertices.size() > 0)
    {
      _vertices = vertices;
    }
    else
    {
      for (int i=0; i<curves.size(); i++)
      {
        _vertices.push_back(make_pair(0,0));
        _curves[i]->value(0, _vertices[i].first, _vertices[i].second);
      }
    }
    if (op==OP_VALUE)
    {
      // if op is not OP_VALUE, we assume that the functions passed in are already bubbles,
      // and that they already run parallel to each other...

      // we assume that the curves go CCW around the element; we flip the two opposite edges
      // so both sets of opposite edges run parallel to each other:
      _curves[2] = ParametricCurve::reverse(_curves[2]);
      _curves[3] = ParametricCurve::reverse(_curves[3]);

      // since we keep _vertices separately, can just store bubble functions in _curves
      _curves[0] = ParametricCurve::bubble(_curves[0]);
      _curves[1] = ParametricCurve::bubble(_curves[1]);
      _curves[2] = ParametricCurve::bubble(_curves[2]);
      _curves[3] = ParametricCurve::bubble(_curves[3]);
    }
    else if (op==OP_DX)
    {
      _curves[0] = _curves[0]->dt_parametric();
      _curves[2] = _curves[2]->dt_parametric();
    }
    else if (op==OP_DY)
    {
      _curves[1] = _curves[1]->dt_parametric();
      _curves[3] = _curves[3]->dt_parametric();
    }
  }
protected:
  TransfiniteInterpolatingSurface(const vector< ParametricCurvePtr > &curves, Camellia::EOperator op,
                                  const vector< pair<double, double> > &vertices)
  {
    init(curves, op, vertices);
  }
public:
  TransfiniteInterpolatingSurface(const vector< ParametricCurvePtr > &curves)
  {
    vector< pair<double, double> > vertices;
    init(curves,OP_VALUE,vertices);
  }
  ParametricCurvePtr edgeBubble(int edgeIndex)
  {
    return _curves[edgeIndex];
  }
  void setNeglectVertices(bool value)
  {
    _neglectVertices = value;
  }
  void value(double t1, double t2, double &x, double &y);
  const vector< pair<double,double> > &vertices()
  {
    return _vertices;
  }
  TFunctionPtr<double> dt1()
  {
    if (_op == OP_VALUE)
    {
      return Teuchos::rcp( new TransfiniteInterpolatingSurface(_curves, OP_DX, _vertices) );
    }
    else
    {
      return Function::null();
    }
  }
  TFunctionPtr<double> dt2()
  {
    if (_op == OP_VALUE)
    {
      return Teuchos::rcp( new TransfiniteInterpolatingSurface(_curves, OP_DY, _vertices) );
    }
    else
    {
      return Function::null();
    }
  }
  TFunctionPtr<double> grad(int numComponents)
  {
    if ((numComponents != -1) && (numComponents != 2))   // -1 means "however many components are available" (2, here)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TransfiniteInterpolatingSurface only supports gradients in 2D.");
    }
    TFunctionPtr<double> parametricGradient = Function::vectorize(dt1(), dt2());
    return ParametricCurve::parametricGradientWrapper(parametricGradient);
  }
  std::vector< TFunctionPtr<double> > memberFunctions()
  {
    std::vector< TFunctionPtr<double> > members;
    for (auto curve : _curves)
    {
      members.push_back(curve);
    }
    return members;
  }
};

void ParametricSurface::basisWeightsForEdgeInterpolant(FieldContainer<double> &edgeInterpolationCoefficients,
    VectorBasisPtr basis,
    MeshPtr mesh, int cellID)
{
  vector< ParametricCurvePtr > curves = mesh->parametricEdgesForCell(cellID);
  Teuchos::RCP<TransfiniteInterpolatingSurface> exactSurface = Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
  exactSurface->setNeglectVertices(false);

  int basisDegree = basis->getDegree();
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  BasisPtr basis1D = BasisFactory::basisFactory()->getBasis(basisDegree, line_2.getKey(),
                     Camellia::FUNCTION_SPACE_HGRAD);

  BasisPtr compBasis = basis->getComponentBasis();
  int numComponents = basis->getNumComponents();
  if (numComponents != 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only 2D surfaces supported right now");
  }

  edgeInterpolationCoefficients.resize(basis->getCardinality());

  set<int> edgeNodeFieldIndices = BasisFactory::basisFactory()->sideFieldIndices(basis,true); // true: include vertex dofs

  FieldContainer<double> dofCoords(compBasis->getCardinality(),2);
  IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >* intrepidBasisWrapper = dynamic_cast< IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >* >(compBasis.get());
  if (!intrepidBasisWrapper)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "compBasis does not appear to be an instance of IntrepidBasisWrapper");
  }
  Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >* intrepidBasis = dynamic_cast< Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >* >(intrepidBasisWrapper->intrepidBasis().get());
  if (!intrepidBasis)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "IntrepidBasisWrapper does not appear to wrap Basis_HGRAD_QUAD_Cn_FEM.");
  }
  intrepidBasis->getDofCoords(dofCoords);

  int edgeDim = 1;
  int vertexDim = 0;

  // set vertex dofs:
  for (int vertexIndex=0; vertexIndex<curves.size(); vertexIndex++)
  {
    double x = exactSurface->vertices()[vertexIndex].first;
    double y = exactSurface->vertices()[vertexIndex].second;
    int compDofOrdinal = compBasis->getDofOrdinal(vertexDim, vertexIndex, 0);
    int basisDofOrdinal_x = basis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 0);
    int basisDofOrdinal_y = basis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 1);
    edgeInterpolationCoefficients[basisDofOrdinal_x] = x;
    edgeInterpolationCoefficients[basisDofOrdinal_y] = y;
  }

  for (int edgeIndex=0; edgeIndex<curves.size(); edgeIndex++)
  {
    bool edgeDofsFlipped = edgeIndex >= 2; // because Intrepid's ordering of dofs on the quad is not CCW but tensor-product, we need to flip for the opposite edges
    // (what makes things worse is that the vertex/edge numbering *is* CCW)
    if (curves.size() != 4)
    {
      cout << "WARNING: have not worked out the rule for flipping or not flipping edge dofs for anything but quads.\n";
    }
    double edgeLength = curves[edgeIndex]->linearLength();

    //    cout << "edgeIndex " << edgeIndex << endl;
    for (int comp=0; comp<numComponents; comp++)
    {
      FieldContainer<double> basisCoefficients_comp;
      bool useH1ForEdgeInterpolant = true; // an experiment
      curves[edgeIndex]->projectionBasedInterpolant(basisCoefficients_comp, basis1D, comp, edgeLength, useH1ForEdgeInterpolant);
      //      cout << "for edge " << edgeIndex << " and comp " << comp << ", projection-based interpolant dofs:\n";
      //      cout << basisCoefficients_comp;
      ////      cout << "basis dof coords:\n" << dofCoords;
      //      int basisDofOrdinal = basis->getDofOrdinalFromComponentDofOrdinal(v0_dofOrdinal_comp, comp);
      //      edgeInterpolationCoefficients[basisDofOrdinal] = basisCoefficients_comp[v0_dofOrdinal_1D];

      if (compBasis->getDegree() >= 2)   // then there are some "middle" nodes on the edge
      {
        // get the first dofOrdinal for the edge, so we can check the number of edge basis functions
        int firstEdgeDofOrdinal = compBasis->getDofOrdinal(edgeDim, edgeIndex, 0);

        //        cout << "first edge dofOrdinal: " << firstEdgeDofOrdinal << endl;

        int numEdgeDofs = compBasis->getDofTag(firstEdgeDofOrdinal)[3];
        if (numEdgeDofs != basis1D->getCardinality() - 2)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numEdgeDofs does not match 1D basis cardinality");
        }
        for (int edgeDofOrdinal=0; edgeDofOrdinal<numEdgeDofs; edgeDofOrdinal++)
        {
          // determine the index into basisCoefficients_comp:
          int edgeDofOrdinalIn1DBasis = edgeDofsFlipped ? numEdgeDofs - 1 - edgeDofOrdinal : edgeDofOrdinal;
          int dofOrdinal1D = basis1D->getDofOrdinal(edgeDim, 0, edgeDofOrdinalIn1DBasis);
          // determine the ordinal of the edge dof in the component basis:
          int compDofOrdinal = compBasis->getDofOrdinal(edgeDim, edgeIndex, edgeDofOrdinal);
          // now, determine its ordinal in the vector basis
          int basisDofOrdinal = basis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, comp);

          //          cout << "edge dof ordinal " << edgeDofOrdinal << " has basis weight " << basisCoefficients_comp[dofOrdinal1D] << " for component " << comp << endl;
          //          cout << "node on cell is at (" << dofCoords(compDofOrdinal,0) << ", " << dofCoords(compDofOrdinal,1) << ")\n";
          //          cout << "mapping to basisDofOrdinal " << basisDofOrdinal << endl;

          edgeInterpolationCoefficients[basisDofOrdinal] = basisCoefficients_comp[dofOrdinal1D];
        }
      }
    }
  }
  edgeInterpolationCoefficients.resize(edgeInterpolationCoefficients.size());

  // print out a report of what the edge interpolation is doing:
  /*cout << "projection-based interpolation of edges maps the following points:\n";
  for (int compDofOrdinal=0; compDofOrdinal<compBasis->getCardinality(); compDofOrdinal++) {
    double x_ref = dofCoords(compDofOrdinal,0);
    double y_ref = dofCoords(compDofOrdinal,1);
    int basisDofOrdinal_x = basis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 0);
    int basisDofOrdinal_y = basis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 1);
    if (edgeNodeFieldIndices.find(basisDofOrdinal_x) != edgeNodeFieldIndices.end()) {
      double x_phys = edgeInterpolationCoefficients[basisDofOrdinal_x];
      double y_phys = edgeInterpolationCoefficients[basisDofOrdinal_y];
      cout << "(" << x_ref << ", " << y_ref << ") --> (" << x_phys << ", " << y_phys << ")\n";
    }
  }*/
}

void ParametricSurface::basisWeightsForProjectedInterpolant(FieldContainer<double> &basisCoefficients, VectorBasisPtr basis,
    MeshPtr mesh, int cellID)
{
  vector< ParametricCurvePtr > curves = mesh->parametricEdgesForCell(cellID);
  Teuchos::RCP<TransfiniteInterpolatingSurface> exactSurface = Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
  exactSurface->setNeglectVertices(false);

  FieldContainer<double> edgeInterpolationCoefficients(basis->getCardinality());
  basisWeightsForEdgeInterpolant(edgeInterpolationCoefficients, basis, mesh, cellID);

  set<int> edgeFieldIndices = BasisFactory::basisFactory()->sideFieldIndices(basis,true); // true: include vertex dofs

  TFunctionPtr<double> edgeInterpolant = Teuchos::rcp( new BasisSumFunction(basis, edgeInterpolationCoefficients) );

  IPPtr L2 = Teuchos::rcp( new IP );
  // we assume that basis is a vector HGRAD basis
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr v = vf->testVar("v", VECTOR_HGRAD);
  L2->addTerm(v);

  IPPtr H1 = Teuchos::rcp( new IP );
//  H1->addTerm(v); // experiment: seminorm is a norm when the edge dofs are excluded--and this is what LD does
  H1->addTerm(v->grad());

  int maxTestDegree = mesh->getElementType(cellID)->testOrderPtr->maxBasisDegree();
  TEUCHOS_TEST_FOR_EXCEPTION(maxTestDegree < 1, std::invalid_argument, "Constant test spaces unsupported.");

  int cubatureDegree = std::max(maxTestDegree*2,15); // chosen to match that used in edge projection.
  
  FieldContainer<double> physicalCellNodes;
  CellTopoPtr cellTopo = mesh->getElementType(cellID)->cellTopoPtr;
  if (cellTopo->getDimension() == 2)
  {
    physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
  }
  if ((cellTopo->getDimension() == 3) && (cellTopo->getTensorialDegree() > 0))
  {
    // then we interpret this as space-time, and we just treat the first temporal side:
    unsigned temporalSideOrdinal = cellTopo->getTemporalSideOrdinal(0);
    FieldContainer<double> spaceTimePhysicalNodes = mesh->physicalCellNodesForCell(cellID);
    int sideDim = cellTopo->getDimension() - 1;
    int nodeCount = cellTopo->getNodeCount(sideDim, temporalSideOrdinal);
    physicalCellNodes.resize(1,nodeCount,sideDim);
    for (int node=0; node<nodeCount; node++)
    {
      int spaceTimeNode = cellTopo->getNodeMap(sideDim, temporalSideOrdinal, node);
      for (int d=0; d<sideDim; d++)
      {
        physicalCellNodes(0,node,d) = spaceTimePhysicalNodes(0,spaceTimeNode,d);
      }
    }
    // replace space-time cell topology with the purely spatial one:
    cellTopo = cellTopo->getSide(temporalSideOrdinal);
  }
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, physicalCellNodes);

  // project, skipping edgeNodeFieldIndices:
  Projector<double>::projectFunctionOntoBasis(basisCoefficients, TFunctionPtr<double>(exactSurface)-edgeInterpolant, basis, basisCache, H1, v, edgeFieldIndices);

  basisCoefficients.resize(basis->getCardinality()); // get rid of dummy numCells dimension
  // add the two sets of basis coefficients together
  for (int i=0; i<edgeInterpolationCoefficients.size(); i++)
  {
    basisCoefficients[i] += edgeInterpolationCoefficients[i];
  }
}

TFunctionPtr<double> ParametricSurface::dt1()
{
  // default: no dt1() defined
  return Function::null();
}

TFunctionPtr<double> ParametricSurface::dt2()
{
  // default: no dt2() defined
  return Function::null();
}

void TransfiniteInterpolatingSurface::value(double t1, double t2, double &x, double &y)
{
  if (_curves.size() == 4)
  {
    // t1 indexes curves 0 and 2, t2 1 and 3
    double x0, y0, x2, y2;
    _curves[0]->value(t1, x0,y0);
    _curves[2]->value(t1, x2,y2);

    double x1, y1, x3, y3;
    _curves[1]->value(t2, x1,y1);
    _curves[3]->value(t2, x3,y3);

    if (_op == OP_VALUE)
    {
      x = x0*(1-t2) + x1 * t1 + x2*t2 + x3*(1-t1);
      y = y0*(1-t2) + y1 * t1 + y2*t2 + y3*(1-t1);
    }
    else if (_op == OP_DX)
    {
      x = x0*(1-t2) + x1 + x2*t2 - x3;
      y = y0*(1-t2) + y1 + y2*t2 - y3;
    }
    else if (_op == OP_DY)
    {
      x = -x0 + x1 * t1 + x2 + x3*(1-t1);
      y = -y0 + y1 * t1 + y2 + y3*(1-t1);
    }

    if (! _neglectVertices)
    {
      if (_op == OP_VALUE)
      {
        x += _vertices[0].first*(1-t1)*(1-t2) + _vertices[1].first*   t1 *(1-t2)
             + _vertices[2].first*   t1*    t2  + _vertices[3].first*(1-t1)*   t2;
        y += _vertices[0].second*(1-t1)*(1-t2) + _vertices[1].second*   t1 *(1-t2)
             + _vertices[2].second*   t1*    t2  + _vertices[3].second*(1-t1)*   t2;
      }
      else if (_op == OP_DX)
      {
        x += -_vertices[0].first*(1-t2) + _vertices[1].first*(1-t2)
             + _vertices[2].first *   t2  - _vertices[3].first*   t2;
        y += -_vertices[0].second*(1-t2) + _vertices[1].second *(1-t2)
             + _vertices[2].second*    t2  - _vertices[3].second *   t2;
      }
      else if (_op == OP_DY)
      {
        x += -_vertices[0].first*(1-t1) - _vertices[1].first*   t1
             + _vertices[2].first*    t1  + _vertices[3].first*(1-t1);
        y += -_vertices[0].second*(1-t1) - _vertices[1].second*   t1
             + _vertices[2].second*    t1  + _vertices[3].second*(1-t1);
      }
    }
  }
  else if (_curves.size() == 3)
  {
    // TODO: implement this
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads and (eventually) triangles supported...");
  }

}

FieldContainer<double> & ParametricSurface::parametricQuadNodes()   // for CellTools cellWorkset argument
{
  static FieldContainer<double> quadNodes(1,4,2);
  static bool quadNodesSet = false;
  // there's probably a cleaner way to statically initialize this container,
  // but this setup should still do so exactly once
  if (!quadNodesSet)
  {
    quadNodes(0,0,0) = 0.0;
    quadNodes(0,0,1) = 0.0;
    quadNodes(0,1,0) = 1.0;
    quadNodes(0,0,1) = 0.0;
    quadNodes(0,2,0) = 1.0;
    quadNodes(0,2,1) = 1.0;
    quadNodes(0,3,0) = 0.0;
    quadNodes(0,3,1) = 1.0;
    quadNodesSet = true;
  }
  return quadNodes;
}

void ParametricSurface::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  FieldContainer<double> parametricPoints = basisCache->computeParametricPoints();
  int numCells = parametricPoints.dimension(0);
  int numPoints = parametricPoints.dimension(1);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    double x, y;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      double t1, t2;
      t1 = parametricPoints(cellIndex,ptIndex,0);
      t2 = parametricPoints(cellIndex,ptIndex,1);
      this->value(t1, t2, x, y);
      values(cellIndex,ptIndex,0) = x;
      values(cellIndex,ptIndex,1) = y;
    }
  }
}

ParametricSurfacePtr ParametricSurface::linearInterpolant(const vector< ParametricCurvePtr > &curves)
{
  return Teuchos::rcp( new LinearInterpolatingSurface(curves) );
}

ParametricSurfacePtr ParametricSurface::transfiniteInterpolant(const vector< ParametricCurvePtr > &curves)
{
  return Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
}
