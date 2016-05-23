#include "MeanShift.h"

#include <stdio.h>
#include <math.h>

#define EPSILON 0.0001

double euclidean_distance(const Eigen::Vector3d &point_a, const Eigen::Vector3d &point_b)
{
    double total = 0;
    Eigen::Vector3d tmp = point_a - point_b;

    /*
    for(int i=0; i<point_a.size(); i++)
    {
        total += (point_a[i]-point_b[i])*(point_a[i]-point_b[i]);
    }
    */

    return tmp.norm();

}

/// Gaussian radial basis function(RBF) kernel, is a popular kernel function used in various kenerlized learning algorithms. In particular, it's commonly used in SVM
/// The RBF kernel on two samples x and x' represented as feature vectors in some input space, is defined as:
///  K(x,x')= exp(-||x-x'||^2/(2*sigma^2))
/// ||x-x'||^2 may be recognized as the squared Euclidean distance between the two feature vectors. sigma is a free parameter. An equivalent, but simpler, definition involves
/// a parameter gamma=1/(2*sigma^2)
/// K(x, x')=exp(-gamma||x-x'||^2)
double gaussian_kernel(double distance, double kernel_bandwidth)
{
    double gauss_kernel = exp(-(distance*distance)/kernel_bandwidth);
    return gauss_kernel;
}

void MeanShift::set_kernel(double (*_kernel_func)(double, double))
{
    kernel_func = gaussian_kernel;
}

Eigen::Vector3d MeanShift::shift_point(const Eigen::Vector3d &point, const vector<Eigen::Vector3d> &points, double kernel_bandwidth)
{
   Eigen::Vector3d shifted_point = point;
   for(int dim=0; dim<shifted_point.size(); dim++)
   {
      shifted_point[dim] = 0;
   }

    double total_weight = 0;
    for(int i=0; i<points.size(); i++)
    {
        Eigen::Vector3d temp_point = points[i];
        double distance = euclidean_distance(point, temp_point);
        double weight = kernel_func(distance, kernel_bandwidth);
        for(int j=0; j<shifted_point.size(); j++)
        {
            shifted_point[j] +=temp_point[j]*weight;
        }
        total_weight +=weight;
    }

    for(int i=0; i<shifted_point.size(); i++)
    {
        shifted_point[i]/=total_weight;
    }

    return shifted_point;
}

vector<Eigen::Vector3d> MeanShift::cluster(vector<Eigen::Vector3d> points, double kernel_bandwidth)
{
    vector<bool> stop_moving(points.size(), false);
    stop_moving.reserve(points.size());
    vector<Eigen::Vector3d> shifted_points = points;
    double max_shift_distance;

    while (max_shift_distance>EPSILON)
    {
        max_shift_distance = 0;
        for(int i=0; i<shifted_points.size(); i++)
        {
            if(!stop_moving[i]) {
                Eigen::Vector3d point_new = shift_point(shifted_points[i], points, kernel_bandwidth);
                double shift_distance = euclidean_distance(point_new, shifted_points[i]);
                if(shift_distance > max_shift_distance)
                {
                    max_shift_distance = shift_distance;
                }

                if(shift_distance <= EPSILON)
                {
                    stop_moving[i] = true;
                }

                shifted_points[i] = point_new;

            }
        }
        /// printf("max_shift_distance: %f\n", max_shift_distance);
    }

    return shifted_points;


}
