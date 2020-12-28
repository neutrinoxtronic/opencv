/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"

namespace cv
{

int Subdiv2D::nextEdge(int edge) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    return qedges[edge >> 2].next[edge & 3];
}

int Subdiv2D::rotateEdge(int edge, int rotate) const
{
    return (edge & ~3) + ((edge + rotate) & 3);
}

int Subdiv2D::symEdge(int edge) const
{
    return edge ^ 2;
}

int Subdiv2D::getEdge(int edge, int nextEdgeType) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    edge = qedges[edge >> 2].next[(edge + nextEdgeType) & 3];
    return (edge & ~3) + ((edge + (nextEdgeType >> 4)) & 3);
}

int Subdiv2D::edgeOrg(int edge, CV_OUT Point2f* orgpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[edge & 3];
    if( orgpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *orgpt = vtx[vidx].pt;
    }
    return vidx;
}

int Subdiv2D::edgeDst(int edge, CV_OUT Point2f* dstpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[(edge + 2) & 3];
    if( dstpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *dstpt = vtx[vidx].pt;
    }
    return vidx;
}


Point2f Subdiv2D::getVertex(int vertex, CV_OUT int* firstEdge) const
{
    CV_DbgAssert((size_t)vertex < vtx.size());
    if( firstEdge )
        *firstEdge = vtx[vertex].firstEdge;
    return vtx[vertex].pt;
}


Subdiv2D::Subdiv2D()
{
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;
}

Subdiv2D::Subdiv2D(Rect rect)
{
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;

    initDelaunay(rect);
}


Subdiv2D::QuadEdge::QuadEdge()
{
    next[0] = next[1] = next[2] = next[3] = 0;
    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

Subdiv2D::QuadEdge::QuadEdge(int edgeidx)
{
    CV_DbgAssert((edgeidx & 3) == 0);
    next[0] = edgeidx;
    next[1] = edgeidx+3;
    next[2] = edgeidx+2;
    next[3] = edgeidx+1;

    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

bool Subdiv2D::QuadEdge::isfree() const
{
    return next[0] <= 0;
}

Subdiv2D::Vertex::Vertex()
{
    firstEdge = 0;
    type = -1;
}

Subdiv2D::Vertex::Vertex(Point2f _pt, bool _isvirtual, int _firstEdge)
{
    firstEdge = _firstEdge;
    type = (int)_isvirtual;
    pt = _pt;
}

bool Subdiv2D::Vertex::isvirtual() const
{
    return type > 0;
}

bool Subdiv2D::Vertex::isfree() const
{
    return type < 0;
}

void Subdiv2D::splice( int edgeA, int edgeB )
{
    int& a_next = qedges[edgeA >> 2].next[edgeA & 3];
    int& b_next = qedges[edgeB >> 2].next[edgeB & 3];
    int a_rot = rotateEdge(a_next, 1);
    int b_rot = rotateEdge(b_next, 1);
    int& a_rot_next = qedges[a_rot >> 2].next[a_rot & 3];
    int& b_rot_next = qedges[b_rot >> 2].next[b_rot & 3];
    std::swap(a_next, b_next);
    std::swap(a_rot_next, b_rot_next);
}

void Subdiv2D::setEdgePoints(int edge, int orgPt, int dstPt)
{
    qedges[edge >> 2].pt[edge & 3] = orgPt;
    qedges[edge >> 2].pt[(edge + 2) & 3] = dstPt;
    vtx[orgPt].firstEdge = edge;
    vtx[dstPt].firstEdge = edge ^ 2;
}

int Subdiv2D::connectEdges( int edgeA, int edgeB )
{
    int edge = newEdge();

    splice(edge, getEdge(edgeA, NEXT_AROUND_LEFT));
    splice(symEdge(edge), edgeB);

    setEdgePoints(edge, edgeDst(edgeA), edgeOrg(edgeB));
    return edge;
}

void Subdiv2D::swapEdges( int edge )
{
    int sedge = symEdge(edge);
    int a = getEdge(edge, PREV_AROUND_ORG);
    int b = getEdge(sedge, PREV_AROUND_ORG);

    splice(edge, a);
    splice(sedge, b);

    setEdgePoints(edge, edgeDst(a), edgeDst(b));

    splice(edge, getEdge(a, NEXT_AROUND_LEFT));
    splice(sedge, getEdge(b, NEXT_AROUND_LEFT));
}

static double distance(Point2f a, Point2f b) {
    return norm(b - a);
}

static bool collinear(Point2f u, Point2f v) {
    return abs(u.cross(v)) < FLT_EPSILON;
}

static int counterClockwise(Point2f a, Point2f b, Point2f c) {
    double cp = (b - a).cross(c - a);
    return cp > FLT_EPSILON ? 1 : (cp < -FLT_EPSILON ? -1 : 0);
}

static int leftOf(Point2f x, Point2f a, Point2f b) {
    return counterClockwise(x, a, b);
}

static int inCircle(Point2f a, Point2f b, Point2f c, Point2f d) {
    const double eps = FLT_EPSILON * 0.125;

    double val =
            ((double)a.x * a.x + (double)a.y * a.y) * ( c - b ).cross( d - b );
    val -=  ((double)b.x * b.x + (double)b.y * b.y) * ( c - a ).cross( d - a );
    val +=  ((double)c.x * c.x + (double)c.y * c.y) * ( b - a ).cross( d - a );
    val -=  ((double)d.x * d.x + (double)d.y * d.y) * ( b - a ).cross( c - a );

    return val > eps ? 1 : val < -eps ? -1 : 0;
}

// the "meta point" predicate
#undef MP
#define MP(i) ((i) == 1 || (i) == 2 || (i) == 3)

int Subdiv2D::counterClockwiseEx(int i, int j, int k) const {

    if (MP(i) && MP(j) && MP(k)) {
        return counterClockwise(vtx[i].pt, vtx[j].pt, vtx[k].pt);
    }

    if (MP(i) && MP(j) && !MP(k)) {
        return counterClockwise(vtx[i].pt, vtx[j].pt, vtx[0].pt);
    }

    if (!MP(i) && !MP(j) && MP(k)) {
        return !collinear(vtx[j].pt - vtx[i].pt, vtx[k].pt) ?
                counterClockwise(vtx[0].pt, vtx[j].pt - vtx[i].pt, vtx[k].pt) :
                counterClockwise(vtx[i].pt, vtx[j].pt, vtx[0].pt);
    }

    if (!MP(i) && !MP(j) && !MP(k)) {
        return counterClockwise(vtx[i].pt, vtx[j].pt, vtx[k].pt);
    }

    return counterClockwiseEx(j, k, i);
}

int Subdiv2D::rightOfEx(int p, int i, int j) const {
    return counterClockwiseEx(p, j, i);
}

int Subdiv2D::inCircleEx(int i, int j, int k, int l) const {

    if (MP(i) && MP(j) && MP(k) && !MP(l)) {
        return counterClockwiseEx(i, j, k);
    }

    if (MP(i) && !MP(j) && MP(k) && !MP(l)) {
        if (!collinear(vtx[l].pt - vtx[j].pt, vtx[k].pt - vtx[i].pt)) {
            return leftOf(vtx[l].pt - vtx[j].pt, vtx[0].pt, vtx[k].pt - vtx[i].pt);
        } else {
            double dl = distance(vtx[0].pt, vtx[l].pt);
            double dj = distance(vtx[0].pt, vtx[j].pt);

            if (abs(dl - dj) < FLT_EPSILON) {
                return 0;
            } else {
                return (dl < dj && counterClockwiseEx(i, j, k) > 0) || (dl > dj && counterClockwiseEx(i, j, k) < 0) ? 1 : -1;
            }
        }
    }

    if (MP(i) && MP(j) && !MP(k) && !MP(l)) {
        return -inCircleEx(i, k, j, l);
    }

    if (!MP(i) && !MP(j) && !MP(k) && MP(l)) {
        if (!collinear(vtx[j].pt - vtx[i].pt, vtx[k].pt - vtx[i].pt)) {
            return -counterClockwise(vtx[i].pt, vtx[j].pt, vtx[k].pt);
        } else {
            return !vtx[k].pt.inside(Rect2f(vtx[i].pt, vtx[j].pt)) ?
                   leftOf(vtx[l].pt, vtx[0].pt, vtx[j].pt - vtx[i].pt) :
                   leftOf(vtx[l].pt, vtx[0].pt, vtx[k].pt - vtx[j].pt);
        }
    }

    if (!MP(i) && !MP(j) && !MP(k) && !MP(l)) {
        return inCircle(vtx[i].pt, vtx[j].pt, vtx[k].pt, vtx[l].pt);
    }

    return -inCircleEx(j, k, l, i);
}

int Subdiv2D::newEdge()
{
    if( freeQEdge <= 0 )
    {
        qedges.push_back(QuadEdge());
        freeQEdge = (int)(qedges.size()-1);
    }
    int edge = freeQEdge*4;
    freeQEdge = qedges[edge >> 2].next[1];
    qedges[edge >> 2] = QuadEdge(edge);
    return edge;
}

void Subdiv2D::deleteEdge(int edge)
{
    CV_DbgAssert((size_t)(edge >> 2) < (size_t)qedges.size());
    splice( edge, getEdge(edge, PREV_AROUND_ORG) );
    int sedge = symEdge(edge);
    splice(sedge, getEdge(sedge, PREV_AROUND_ORG) );

    edge >>= 2;
    qedges[edge].next[0] = 0;
    qedges[edge].next[1] = freeQEdge;
    freeQEdge = edge;
}

int Subdiv2D::newPoint(Point2f pt, bool isvirtual, int firstEdge, bool isfree)
{
    if( freePoint == 0 )
    {
        vtx.push_back(Vertex());
        freePoint = (int)(vtx.size()-1);
    }
    int vidx = freePoint;
    freePoint = vtx[vidx].firstEdge;
    vtx[vidx] = Vertex(pt, isvirtual, firstEdge);
    if (isfree) {
        vtx[vidx].type = -1;
    }

    return vidx;
}

void Subdiv2D::deletePoint(int vidx)
{
    CV_DbgAssert( (size_t)vidx < vtx.size() );
    vtx[vidx].firstEdge = freePoint;
    vtx[vidx].type = -1;
    freePoint = vidx;
}

int Subdiv2D::locate(Point2f pt, int& _edge, int& _vertex)
{
    CV_INSTRUMENT_REGION();

    int vertex = 0;

    int i, maxEdges = (int)(qedges.size() * 4);

    if( qedges.size() < (size_t)4 )
        CV_Error( CV_StsError, "Subdivision is empty" );

    if( pt.x < topLeft.x || pt.y < topLeft.y || pt.x >= bottomRight.x || pt.y >= bottomRight.y )
        CV_Error( CV_StsOutOfRange, "" );

    int edge = recentEdge;
    CV_Assert(edge > 0);

    int location = PTLOC_ERROR;

    int curr_point = newPoint(pt, false, 0, true);
    int right_of_curr = rightOfEx(curr_point, edgeOrg(edge), edgeDst(edge));
    if( right_of_curr > 0 )
    {
        edge = symEdge(edge);
        right_of_curr = -right_of_curr;
    }

    for( i = 0; i < maxEdges; i++ )
    {
        int onext_edge = nextEdge( edge );
        int dprev_edge = getEdge( edge, PREV_AROUND_DST );

        int right_of_onext = rightOfEx(curr_point, edgeOrg(onext_edge), edgeDst(onext_edge));
        int right_of_dprev = rightOfEx(curr_point, edgeOrg(dprev_edge), edgeDst(dprev_edge));

        if( right_of_dprev > 0 )
        {
            if( right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) )
            {
                location = PTLOC_INSIDE;
                break;
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
        else
        {
            if( right_of_onext > 0 )
            {
                if( right_of_dprev == 0 && right_of_curr == 0 )
                {
                    location = PTLOC_INSIDE;
                    break;
                }
                else
                {
                    right_of_curr = right_of_dprev;
                    edge = dprev_edge;
                }
            }
            else if( right_of_curr == 0 &&
                    rightOfEx(edgeDst(onext_edge), edgeOrg(edge), edgeDst(edge)) >= 0 )
            {
                edge = symEdge( edge );
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
    }

    recentEdge = edge;

    if( location == PTLOC_INSIDE )
    {
        Point2f org_pt, dst_pt;
        edgeOrg(edge, &org_pt);
        edgeDst(edge, &dst_pt);

        double t1 = fabs( pt.x - org_pt.x );
        t1 += fabs( pt.y - org_pt.y );
        double t2 = fabs( pt.x - dst_pt.x );
        t2 += fabs( pt.y - dst_pt.y );
        double t3 = fabs( org_pt.x - dst_pt.x );
        t3 += fabs( org_pt.y - dst_pt.y );

        if( t1 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeOrg( edge );
            edge = 0;
        }
        else if( t2 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeDst( edge );
            edge = 0;
        }
        else if((t1 < t3 || t2 < t3) &&
                collinear(org_pt - pt, dst_pt - pt))
        {
            location = PTLOC_ON_EDGE;
            vertex = 0;
        }
    }

    if( location == PTLOC_ERROR )
    {
        edge = 0;
        vertex = 0;
    }

    deletePoint(curr_point);

    _edge = edge;
    _vertex = vertex;

    return location;
}

int Subdiv2D::insert(Point2f pt)
{
    CV_INSTRUMENT_REGION();

    int curr_point = 0, curr_edge = 0, deleted_edge = 0;
    int location = locate( pt, curr_edge, curr_point );

    if( location == PTLOC_ERROR )
        CV_Error( CV_StsBadSize, "" );

    if( location == PTLOC_OUTSIDE_RECT )
        CV_Error( CV_StsOutOfRange, "" );

    if( location == PTLOC_VERTEX )
        return curr_point;

    if( location == PTLOC_ON_EDGE )
    {
        deleted_edge = curr_edge;
        recentEdge = curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        deleteEdge(deleted_edge);
    }
    else if( location == PTLOC_INSIDE )
        ;
    else
        CV_Error_(CV_StsError, ("Subdiv2D::locate returned invalid location = %d", location) );

    assert( curr_edge != 0 );
    validGeometry = false;

    curr_point = newPoint(pt, false);
    int base_edge = newEdge();
    int first_point = edgeOrg(curr_edge);
    setEdgePoints(base_edge, first_point, curr_point);
    splice(base_edge, curr_edge);

    do
    {
        base_edge = connectEdges( curr_edge, symEdge(base_edge) );
        curr_edge = getEdge(base_edge, PREV_AROUND_ORG);
    }
    while( edgeDst(curr_edge) != first_point );

    curr_edge = getEdge( base_edge, PREV_AROUND_ORG );

    int i, max_edges = (int)(qedges.size()*4);

    for( i = 0; i < max_edges; i++ )
    {
        int temp_dst = 0, curr_org = 0, curr_dst = 0;
        int temp_edge = getEdge( curr_edge, PREV_AROUND_ORG );

        temp_dst = edgeDst( temp_edge );
        curr_org = edgeOrg( curr_edge );
        curr_dst = edgeDst( curr_edge );

        if(rightOfEx(temp_dst, edgeOrg(curr_edge), edgeDst(curr_edge)) > 0 &&
                inCircleEx(curr_org, temp_dst, curr_dst, curr_point) > 0 )
        {
            swapEdges( curr_edge );
            curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        }
        else if( curr_org == first_point )
            break;
        else
            curr_edge = getEdge( nextEdge( curr_edge ), PREV_AROUND_LEFT );
    }

    return curr_point;
}

void Subdiv2D::insert(const std::vector<Point2f>& ptvec)
{
    CV_INSTRUMENT_REGION();

    for( size_t i = 0; i < ptvec.size(); i++ )
        insert(ptvec[i]);
}

void Subdiv2D::initDelaunay( Rect rect )
{
    CV_INSTRUMENT_REGION();

    float rx = (float)rect.x;
    float ry = (float)rect.y;

    vtx.clear();
    qedges.clear();

    recentEdge = 0;
    validGeometry = false;

    topLeft = Point2f( rx, ry );
    bottomRight = Point2f( rx + rect.width, ry + rect.height );

    Point2f ppA( 1.f, 0.f );
    Point2f ppB( cos(2.f * (float)M_PI / 3.f), sin(2.f * (float)M_PI / 3.f) );
    Point2f ppC( cos(4.f * (float)M_PI / 3.f), sin(4.f * (float)M_PI / 3.f) );

    vtx.push_back(Vertex());
    qedges.push_back(QuadEdge());

    freeQEdge = 0;
    freePoint = 0;

    int pA = newPoint(ppA, false);
    int pB = newPoint(ppB, false);
    int pC = newPoint(ppC, false);

    int edge_AB = newEdge();
    int edge_BC = newEdge();
    int edge_CA = newEdge();

    setEdgePoints( edge_AB, pA, pB );
    setEdgePoints( edge_BC, pB, pC );
    setEdgePoints( edge_CA, pC, pA );

    splice( edge_AB, symEdge( edge_CA ));
    splice( edge_BC, symEdge( edge_AB ));
    splice( edge_CA, symEdge( edge_BC ));

    recentEdge = edge_AB;
}


void Subdiv2D::clearVoronoi()
{
    size_t i, total = qedges.size();

    for( i = 0; i < total; i++ )
        qedges[i].pt[1] = qedges[i].pt[3] = 0;

    total = vtx.size();
    for( i = 0; i < total; i++ )
    {
        if( vtx[i].isvirtual() )
            deletePoint((int)i);
    }

    validGeometry = false;
}


static Point2f computeVoronoiPoint(Point2f org0, Point2f dst0, Point2f org1, Point2f dst1)
{
    double a0 = dst0.x - org0.x;
    double b0 = dst0.y - org0.y;
    double c0 = -0.5*(a0 * (dst0.x + org0.x) + b0 * (dst0.y + org0.y));

    double a1 = dst1.x - org1.x;
    double b1 = dst1.y - org1.y;
    double c1 = -0.5*(a1 * (dst1.x + org1.x) + b1 * (dst1.y + org1.y));

    double det = a0 * b1 - a1 * b0;

    if( det != 0 )
    {
        det = 1. / det;
        return Point2f((float) ((b0 * c1 - b1 * c0) * det),
                       (float) ((a1 * c0 - a0 * c1) * det));
    }

    return Point2f(FLT_MAX, FLT_MAX);
}


void Subdiv2D::calcVoronoi()
{
    // check if it is already calculated
    if( validGeometry )
        return;

    clearVoronoi();

    Point2f topRight(bottomRight.x, topLeft.y), bottomLeft(topLeft.x, bottomRight.y);
    double radius = max(
            max(distance(vtx[0].pt, bottomRight), distance(vtx[0].pt, topLeft)),
            max(distance(vtx[0].pt, topRight), distance(vtx[0].pt, bottomLeft)));

    for ( int quad_edge = 1; quad_edge < 4; quad_edge++) {
        int edge0 = quad_edge * 4;
        Point2f org0, dst0;
        edgeOrg(edge0, &org0);
        edgeDst(edge0, &dst0);

        int edge1 = getEdge(edge0, NEXT_AROUND_LEFT);
        int edge2 = getEdge(edge1, NEXT_AROUND_LEFT);

        int edge3 = getEdge(edge1, NEXT_AROUND_DST);
        Point2f org3, dst3;
        if (!MP(edgeOrg(edge3, &org3)) && !MP(edgeDst(edge3, &dst3))) {
            Point2f pt = computeVoronoiPoint(org0, dst0, org3, dst3);
            if (pt.x < FLT_MAX && pt.y < FLT_MAX) {
                if (leftOf(pt, vtx[0].pt, dst0 - org0) < 0 && leftOf(pt, org3, dst3) > 0) {
                    radius = max(radius, distance(vtx[0].pt, pt));
                }
            }
        }

        int edge4 = getEdge(edge2, PREV_AROUND_ORG);
        Point2f org4, dst4;
        if (!MP(edgeOrg(edge4, &org4)) && !MP(edgeDst(edge4, &dst4))) {
            Point2f pt = computeVoronoiPoint(org0, dst0, org4, dst4);
            if (pt.x < FLT_MAX && pt.y < FLT_MAX) {
                if (leftOf(pt, vtx[0].pt, dst0 - org0) < 0 && leftOf(pt, org4, dst4) > 0) {
                    radius = max(radius, distance(vtx[0].pt, pt));
                }
            }
        }
    }

    // loop through all quad-edges, except for the first 3 (#1, #2, #3 - 0 is reserved for "NULL" pointer)
    for( int quad_edge = 4; quad_edge < (int)qedges.size(); ++quad_edge ) {
        if (qedges[quad_edge].isfree()) {
            continue;
        }

        for ( int i = 0, edge0 = quad_edge * 4; i < 2; ++i, edge0 = symEdge(edge0) ) {
            if (!qedges[edge0 >> 2].pt[3 - (edge0 & 2)]) {
                int edge1 = getEdge( edge0, NEXT_AROUND_LEFT );
                int edge2 = getEdge( edge1, NEXT_AROUND_LEFT );

                Point2f org0, dst0, dst1;
                if (!MP(edgeOrg(edge0, &org0)) && !MP(edgeDst(edge0, &dst0)) && !MP(edgeDst(edge1, &dst1))) {
                    Point2f virt_point = computeVoronoiPoint(org0, dst0, dst0, dst1);
                    qedges[edge0 >> 2].pt[3 - (edge0 & 2)] =
                    qedges[edge1 >> 2].pt[3 - (edge1 & 2)] =
                    qedges[edge2 >> 2].pt[3 - (edge2 & 2)] = newPoint(virt_point, true);

                    radius = max(radius, distance(vtx[0].pt, virt_point));
                }
            }
        }
    }

    for( int quad_edge = 4; quad_edge < (int)qedges.size(); ++quad_edge ) {
        if (qedges[quad_edge].isfree()) {
            continue;
        }

        for ( int i = 0, edge0 = quad_edge * 4; i < 2; ++i, edge0 = symEdge(edge0) ) {
            if (!qedges[edge0 >> 2].pt[3 - (edge0 & 2)]) {
                int edge1 = getEdge(edge0, NEXT_AROUND_LEFT);
                int edge2 = getEdge(edge1, NEXT_AROUND_LEFT);

                Point2f org0, dst0, dst1;
                if (MP(edgeOrg(edge0, &org0))) {
                    org0 *= 3.f * (float)radius;
                }
                if (MP(edgeDst(edge0, &dst0))) {
                    dst0 *= 3.f * (float)radius;
                }
                if (MP(edgeDst(edge1, &dst1))) {
                    dst1 *= 3.f * (float)radius;
                }

                Point2f virt_point = computeVoronoiPoint(org0, dst0, dst0, dst1);
                qedges[edge0 >> 2].pt[3 - (edge0 & 2)] =
                qedges[edge1 >> 2].pt[3 - (edge1 & 2)] =
                qedges[edge2 >> 2].pt[3 - (edge2 & 2)] = newPoint(virt_point, true);
            }
        }
    }

    validGeometry = true;
}


int Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt)
{
    CV_INSTRUMENT_REGION();

    if( !validGeometry )
        calcVoronoi();

    int vertex = 0, edge = 0;
    int loc = locate( pt, edge, vertex );

    if( loc != PTLOC_ON_EDGE && loc != PTLOC_INSIDE )
        return vertex;

    vertex = 0;

    int curr_point = newPoint(pt, false, 0, true);

    int start = edgeOrg(edge);

    edge = rotateEdge(edge, 1);

    int i, total = (int)vtx.size();

    for( i = 0; i < total; i++ )
    {

        for(;;)
        {
            CV_Assert( edgeDst( edge ) > 0 );
            if(rightOfEx(curr_point, start, edgeDst(edge)) >= 0 )
                break;

            edge = getEdge( edge, NEXT_AROUND_LEFT );
        }

        for(;;)
        {
            CV_Assert( edgeOrg( edge ) > 0 );

            if(rightOfEx(curr_point, start, edgeOrg(edge)) < 0 )
                break;

            edge = getEdge( edge, PREV_AROUND_LEFT );
        }

        if(rightOfEx(curr_point, edgeDst(edge), edgeOrg(edge)) >= 0 )
        {
            vertex = edgeOrg(rotateEdge( edge, 3 ));
            break;
        }

        edge = symEdge( edge );
    }

    deletePoint(curr_point);

    if( nearestPt && vertex > 0 )
        *nearestPt = vtx[vertex].pt;

    return vertex;
}

void Subdiv2D::getEdgeList(std::vector<Vec4f>& edgeList) const
{
    edgeList.clear();

    for( size_t i = 4; i < qedges.size(); i++ )
    {
        if( qedges[i].isfree() )
            continue;
        if( qedges[i].pt[0] > 0 && qedges[i].pt[2] > 0 )
        {
            Point2f org = vtx[qedges[i].pt[0]].pt;
            Point2f dst = vtx[qedges[i].pt[2]].pt;
            edgeList.push_back(Vec4f(org.x, org.y, dst.x, dst.y));
        }
    }
}

void Subdiv2D::getLeadingEdgeList(std::vector<int>& leadingEdgeList) const
{
    leadingEdgeList.clear();
    int i, total = (int)(qedges.size()*4);
    std::vector<bool> edgemask(total, false);

    for( i = 4; i < total; i += 2 )
    {
        if( edgemask[i] )
            continue;
        int edge = i;
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgemask[edge] = true;
        leadingEdgeList.push_back(i);
    }
}

void Subdiv2D::getTriangleList(std::vector<Vec6f>& triangleList) const
{
    triangleList.clear();
    int i, total = (int)(qedges.size()*4);
    std::vector<bool> edgemask(total, false);
    Rect2f rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

    for( i = 4; i < total; i += 2 )
    {
        if( edgemask[i] ) {
            continue;
        }

        Point2f a, b, c;
        int edge_a = i;
        if (MP(edgeOrg(edge_a, &a))) {
            continue;
        }
        int edge_b = getEdge(edge_a, NEXT_AROUND_LEFT);
        if (MP(edgeOrg(edge_b, &b))) {
            continue;
        }
        int edge_c = getEdge(edge_b, NEXT_AROUND_LEFT);
        if (MP(edgeOrg(edge_c, &c))) {
            continue;
        }
        edgemask[edge_a] = true;
        edgemask[edge_b] = true;
        edgemask[edge_c] = true;
        triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
    }
}

void Subdiv2D::getVoronoiFacetList(const std::vector<int>& idx,
                                   CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                   CV_OUT std::vector<Point2f>& facetCenters)
{
    calcVoronoi();
    facetList.clear();
    facetCenters.clear();

    std::vector<Point2f> buf;

    size_t i, total;
    if( idx.empty() )
        i = 4, total = vtx.size();
    else
        i = 0, total = idx.size();

    for( ; i < total; i++ )
    {
        int k = idx.empty() ? (int)i : idx[i];

        if( vtx[k].isfree() || vtx[k].isvirtual() )
            continue;
        int edge = rotateEdge(vtx[k].firstEdge, 1), t = edge;

        // gather points
        buf.clear();
        do
        {
            buf.push_back(vtx[edgeOrg(t)].pt);
            t = getEdge( t, NEXT_AROUND_LEFT );
        }
        while( t != edge );

        facetList.push_back(buf);
        facetCenters.push_back(vtx[k].pt);
    }
}


void Subdiv2D::checkSubdiv() const
{
    int i, j, total = (int)qedges.size();

    for( i = 0; i < total; i++ )
    {
        const QuadEdge& qe = qedges[i];

        if( qe.isfree() )
            continue;

        for( j = 0; j < 4; j++ )
        {
            int e = (int)(i*4 + j);
            int o_next = nextEdge(e);
            int o_prev = getEdge(e, PREV_AROUND_ORG );
            int d_prev = getEdge(e, PREV_AROUND_DST );
            int d_next = getEdge(e, NEXT_AROUND_DST );

            // check points
            CV_Assert( edgeOrg(e) == edgeOrg(o_next));
            CV_Assert( edgeOrg(e) == edgeOrg(o_prev));
            CV_Assert( edgeDst(e) == edgeDst(d_next));
            CV_Assert( edgeDst(e) == edgeDst(d_prev));

            if( j % 2 == 0 )
            {
                CV_Assert( edgeDst(o_next) == edgeOrg(d_prev));
                CV_Assert( edgeDst(o_prev) == edgeOrg(d_next));
                CV_Assert( getEdge(getEdge(getEdge(e,NEXT_AROUND_LEFT),NEXT_AROUND_LEFT),NEXT_AROUND_LEFT) == e );
                CV_Assert( getEdge(getEdge(getEdge(e,NEXT_AROUND_RIGHT),NEXT_AROUND_RIGHT),NEXT_AROUND_RIGHT) == e);
            }
        }
    }
}

}

/* End of file. */
