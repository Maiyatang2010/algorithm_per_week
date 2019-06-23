#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


// kd-point
struct coordinate
{
    double x = 0;
    double y = 0;
    coordinate(){}
    coordinate(double x_, double y_): x(x_), y(y_){}
};

// kd tree node
struct TreeNode
{
    struct coordinate splitor;
    size_t split_dim=0;
    struct TreeNode *left=nullptr;
    struct TreeNode *right=nullptr;
};

// x-sort
bool cmp1(const coordinate& a, const coordinate& b)
{
    return a.x < b.x;
}

// y-sort
bool cmp2(const coordinate& a, const coordinate& b)
{
    return a.y < b.y;
}

// equal
bool equal(const coordinate& a, const coordinate& b)
{
    return a.x == b.x && a.y == b.y;
}

// choose split pointer
void ChooseSplitor(vector<coordinate> &points, size_t &dim, coordinate &splitor)
{
    // variance-x
    double tmp1=0, tmp2=0;
    int size = points.size();
    for(auto point : points)
    {
        tmp1 += 1.0 / double(size) * point.x * point.x;
        tmp2 += 1.0 / double(size) * point.x;
    }
    double var_x = tmp1 - tmp2 * tmp2;

    // variance-y
    tmp1 = tmp2 = 0;
    for(auto point : points)
    {
        tmp1 += 1.0 / size * point.y * point.y;
        tmp2 += 1.0 / size * point.y;
    }
    double var_y = tmp1 - tmp2 * tmp2;

    dim = var_x > var_y ? 0 : 1;
    if(dim == 0)
        sort(points.begin(), points.end(), cmp1);
    else
        sort(points.begin(), points.end(), cmp2);

    splitor.x = points[size/2].x;
    splitor.y = points[size/2].y;
}


TreeNode* build_kdtree(vector<coordinate> &points)
{
    int size = points.size();
    if(size == 0)
        return nullptr;
    else
    {
        // find splitor
        size_t dim;            // split dimension
        struct coordinate splitor;     // split point
        ChooseSplitor(points, dim, splitor);


        // point distribute
        vector<coordinate> points_left;
        vector<coordinate> points_right;
        if(dim == 0)
        {
            for(auto point : points)
            {
                if(!equal(point, splitor) && point.x <= splitor.x)
                    points_left.push_back(point);
                else if(!equal(point, splitor) && point.x > splitor.x)
                    points_right.push_back(point);
            }
        }
        else
        {
            for(auto point : points)
            {
                if(!equal(point, splitor) && point.y <= splitor.y)
                    points_left.push_back(point);
                else if(!equal(point, splitor) && point.y > splitor.y)
                    points_right.push_back(point);
            }
        }

        // recursive split
        struct TreeNode *T = new TreeNode;
        (T->splitor).x = splitor.x;
        (T->splitor).y = splitor.y;
        T->split_dim = dim;
        T->left = build_kdtree(points_left);
        T->right = build_kdtree(points_right);
        return T;
    }
}

int main(int argc, char* argv[])
{
    vector<coordinate> points = {
            coordinate(2, 3),
            coordinate(5, 4),
            coordinate(9, 6),
            coordinate(4, 7),
            coordinate(8, 1),
            coordinate(7, 2)
    };
    struct TreeNode * root = build_kdtree(points);
    cout<< (root->splitor).x << (root->splitor).y <<endl;
}