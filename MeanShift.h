#pragma once

/*
  MeanShift implementation from https://github.com/mattnedrich/MeanShift_cpp
*/

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <vector>
#include <bitset>
#include <Eigen/Geometry>

using namespace std;

///http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TUZEL1/MeanShift.pdf
///The mean shift algorithm is a nonparametric clustering technique which does not require prior knowledge of the number of clusters, and does not constrain the shape of the clusters
///Given n data points x_i, i=1, ... ,n on a d-dimensional space R^D,, the multivariate kernel density estimate obtained with kernel K(x) and widnow radius h is
/// f(x)=(1/nh^d)sum[k(x-x_i)/h]
/// For radially symmetric kernels, it suffices to define the profile of the kernel k(x) satisfying
/// K(x)=c_(k,d)k(||x||^2)
/// where c_(k,d) is a noramlization constant which assured K(x) integrates to 1. The modes of the density function are located at the zeros of the gradient function
/// Gradient f(x)=0.
/// The second term of the gradient is the mean shift. The mean shift vector always points toward the direction of the maximum increase in the density.
/// The mean shift procedure, obtained by successive
/// 1. computation of the mean shift vector m_h(x^t)
/// translation of the window x^(t+1)=x^t+m_h(x^t)
/// is guaranteed to converge to a point where the gradient of density function is zero. Mean Shift mode finding process

class MeanShift
{
    public:
        ///MeanShift constructor takes a function pointer to a kernel function to be used in the clustering process. If NULL, it will use a Gaussian Kernel
        MeanShift() {set_kernel(NULL); }

        MeanShift(double (*_kernel_func)(double, double)) {set_kernel(kernel_func);}

        vector<Eigen::Vector3d> cluster(vector<Eigen::Vector3d>, double);

    private:
        double (*kernel_func)(double, double);
        void set_kernel(double (*_kernel_func)(double, double));
        Eigen::Vector3d shift_point(const Eigen::Vector3d &, const vector<Eigen::Vector3d> &, double);
};

#endif // MEANSHIFT_H
