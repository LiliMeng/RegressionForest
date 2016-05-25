//
//  RGBGUtil.cpp
//  LoopClosure
//
//  Authors: jimmy and Lili on 2016-04-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "RGBGUtil.hpp"
#include "cvx_io.hpp"
#include <string>
#include <unordered_map>

using std::string;
using std::unordered_map;
using cv::Mat;

void RGBGUtil::mean_stddev(const vector<RGBGLearningSample> & samples,
                           const vector<unsigned int> & indices,
                           cv::Point3d & mean_pt,
                           cv::Vec3d & stddev)
{
    assert(indices.size() > 0);

    mean_pt = cv::Point3d(0, 0, 0);
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        mean_pt += samples[index].p3d_;
    }

    //mean_pt /= (double)indices.size();
    mean_pt.x = (double)mean_pt.x/(double)indices.size();
    mean_pt.y = (double)mean_pt.y/(double)indices.size();
    mean_pt.z = (double)mean_pt.z/(double)indices.size();
    double devx = 0.0;
    double devy = 0.0;
    double devz = 0.0;
    for (int i = 0 ; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        cv::Point3d dif = samples[index].p3d_ - mean_pt;
        devx += dif.x * dif.x;
        devy += dif.y * dif.y;
        devz += dif.z * dif.z;
    }
    devx = sqrt(devx/indices.size());
    devy = sqrt(devy/indices.size());
    devz = sqrt(devz/indices.size());
    stddev = cv::Vec3d(devx, devy, devz);
    return;
}

void RGBGUtil::mean_stddev(const vector<cv::Point3d> & points, cv::Point3d & mean_pos, cv::Vec3d & std_pos)
{
    assert(points.size() > 0);

    const int N = (int)points.size();

    // mean position
    mean_pos = cv::Point3d(0, 0, 0);
    for (int i = 0; i<points.size(); i++) {
        mean_pos += points[i];
    }
    //mean_pos /= (double)N;
    mean_pos.x=(double)mean_pos.x/(double)N;
    mean_pos.y=(double)mean_pos.y/(double)N;
    mean_pos.z=(double)mean_pos.z/(double)N;

    // standard deviation
    double dev_x = 0.0;
    double dev_y = 0.0;
    double dev_z = 0.0;
    for (int i = 0; i<points.size(); i++) {
        cv::Point3d dif = mean_pos - points[i];
        dev_x += dif.x * dif.x;
        dev_y += dif.y * dif.y;
        dev_z += dif.z * dif.z;
    }

    dev_x = sqrt(dev_x/N);
    dev_y = sqrt(dev_y/N);
    dev_z = sqrt(dev_z/N);

    std_pos = cv::Vec3d(dev_x, dev_y, dev_z);
}

void RGBGUtil::mean_stddev(const vector<cv::Vec3d> & data,
                           cv::Vec3d & mean,
                           cv::Vec3d & stddev)
{
    assert(data.size() > 0);

    mean = cv::Vec3d(0, 0, 0);
    for (int i = 0; i<data.size(); i++) {
        mean += data[i];
    }
    mean /= (double)data.size();

    stddev = cv::Vec3d(0, 0, 0);
    for (int i = 0; i<data.size(); i++) {
        cv::Vec3d dif = data[i] - mean;
        stddev[0] += dif[0] * dif[0];
        stddev[1] += dif[1] * dif[1];
        stddev[2] += dif[2] * dif[2];
    }

    for (int i = 0; i<3; i++) {
        stddev[i] = sqrt(stddev[i]/(double)data.size());
    }
}

///Revised by Lili
///V(S) in 2.4 CVPR2013
double RGBGUtil::spatial_variance(const vector<RGBGLearningSample> & samples, const vector<unsigned int> & indices)
{
    cv::Point3d mean_pt = cv::Point3d(0, 0, 0);

    for(int i=0; i<indices.size(); i++)
    {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        mean_pt += samples[index].p3d_;
    }

    mean_pt.x = (double)mean_pt.x/(double)indices.size();
    mean_pt.y = (double)mean_pt.y/(double)indices.size();
    mean_pt.z = (double)mean_pt.z/(double)indices.size();

    double var = 0.0;
    for (int i=0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        cv::Point3d dif = mean_pt - samples[index].p3d_;
        var += (diff.x* diff.x + diff.y*diff.y + dif.z*dif.z);
    }

    double var_final = var *(1.0f/(double)indices.size());

    return var_final;
   /*
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        cv::Point3d dif = mean_pt - samples[index].p3d_;
        var += dif.x * dif.x;
        var += dif.y * dif.y;
        var += dif.z * dif.z;
    }
    //var /= indices.size();
    return var;
    */
}

vector<RGBGLearningSample>
RGBGUtil::randomSampleFromRgbdImagesWithoutDepth(const char * rgb_img_file,
                                                 const char * depth_img_file,
                                                 const char * camera_pose_file,
                                                 const int num_sample,
                                                 const int image_index,
                                                 const bool use_depth,
                                                 const bool verbose)
{
    assert(rgb_img_file);
    assert(depth_img_file);
    assert(camera_pose_file);

    vector<RGBGLearningSample> samples;

    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = cvx_io::imread_depth_16bit_to64f(depth_img_file, camera_depth_img);
    assert(is_read);
    cvx_io::imread_rgb_8u(rgb_img_file, rgb_img);

//  cv::imshow("before blur", rgb_img);
//  cv::GaussianBlur(rgb_img, rgb_img, cv::Size(5, 5), 0.0);
//  cv::imshow("after blur", rgb_img);
//  cv::waitKey();

    cv::Mat pose = ms_7_scenes_util::read_pose_7_scenes(camera_pose_file);



}
vector<RGBGLearningSample>
RGBGUtil::randomSampleFromRgbdImagesWithoutDepth(const char * rgb_img_file,
                                                 const char * depth_img_file,
                                                 const char * camera_pose_file,
                                                 const int num_sample,
                                                 const int image_index,
                                                 const bool use_depth,
                                                 const bool verbose)
{
    assert(rgb_img_file);
    assert(depth_img_file);
    assert(camera_pose_file);

    vector<RGBGLearningSample> samples;

    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = cvx_io::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    cvx_io::imread_rgb_8u(rgb_img_file, rgb_img);

 //   cv::imshow("before blur", rgb_img);
 //   cv::GaussianBlur(rgb_img, rgb_img, cv::Size(5, 5), 0.0);
 //   cv::imshow("after blur", rgb_img);
 //   cv::waitKey();


    cv::Mat pose = ms_7_scenes_util::read_pose_7_scenes(camera_pose_file);

    const int width = rgb_img.cols;
    const int height = rgb_img.rows;

    cv::Mat mask;
    cv::Mat world_coordinate = ms_7_scenes_util::camera_depth_to_world_coordinate(camera_depth_img, pose, mask);

    vector<double> pos_x;
    vector<double> pos_y;
    vector<double> pos_z;
    for (int i = 0; i<num_sample; i++) {
        int x = rand()%width;
        int y = rand()%height;

        // ignore bad depth point
        if (mask.at<unsigned char>(y, x) == 0) {
            continue;
        }
        double depth = 1.0;
        if (use_depth) {
            depth = camera_depth_img.at<double>(y, x)/1000.0;
        }
        RGBGLearningSample sp;
        sp.p2d_ = cv::Vec2i(x, y);
        sp.inv_depth_ = 1.0/depth;
        sp.image_index_ = image_index;
        sp.p3d_.x = world_coordinate.at<cv::Vec3d>(y, x)[0];
        sp.p3d_.y = world_coordinate.at<cv::Vec3d>(y, x)[1];
        sp.p3d_.z = world_coordinate.at<cv::Vec3d>(y, x)[2];
        sp.color_[0] = rgb_img.at<cv::Vec3b>(y, x)[0];
        sp.color_[1] = rgb_img.at<cv::Vec3b>(y, x)[1];
        sp.color_[2] = rgb_img.at<cv::Vec3b>(y, x)[2];


        samples.push_back(sp);

        // only for debug
        pos_x.push_back(sp.p3d_.x);
        pos_y.push_back(sp.p3d_.y);
        pos_z.push_back(sp.p3d_.z);
    }
    if (verbose) {
        printf("rgb image is %s\n", rgb_img_file);
        printf("depth image is %s\n", depth_img_file);
        printf("camera pose file is %s\n", camera_pose_file);
    }


    if (0) {
        // only for debug
        double x_min = *std::min_element(pos_x.begin(), pos_x.end());
        double x_max = *std::max_element(pos_x.begin(), pos_x.end());
        double x_range = x_max - x_min;

        double y_min = *std::min_element(pos_y.begin(), pos_y.end());
        double y_max = *std::max_element(pos_y.begin(), pos_y.end());
        double y_range = y_max - y_min;

        double z_min = *std::min_element(pos_z.begin(), pos_z.end());
        double z_max = *std::max_element(pos_z.begin(), pos_z.end());
        double z_range = z_max - z_min;

        printf("min, max x are: %lf %lf %lf\n", x_min, x_max, x_range);
        printf("min, max y are: %lf %lf %lf\n", y_min, y_max, y_range);
        printf("min, max z are: %lf %lf %lf\n", z_min, z_max, z_range);

        // normalize x, y, z to [0,1]
        for (int i = 0; i<samples.size(); i++) {
            RGBGLearningSample sp = samples[i];
            sp.p3d_.x -= x_min;
            sp.p3d_.x /= x_range;

            sp.p3d_.y -= y_min;
            sp.p3d_.y /= y_range;

            sp.p3d_.z -= z_min;
            sp.p3d_.z /= z_range;
            samples[i] = sp;
        }
    }

    if(verbose) {
        printf("sampled %lu samples\n", samples.size());
    }
    return samples;
}

vector<RGBGLearningSample> RGBGUtil::randomSampleFromRgbWithScale(const char * rgb_img_file,
                                                                                     const char * depth_img_file,
                                                                                     const char * camera_pose_file,
                                                                                     const int num_sample,
                                                                                     const int image_index,
                                                                                     const double scale,
                                                                                     cv::Mat & scaled_rgb_img)
{
    vector<RGBGLearningSample> samples;

    // read depth image and RGB image
    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = cvx_io::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    cvx_io::imread_rgb_8u(rgb_img_file, rgb_img);

    cv::Mat pose = ms_7_scenes_util::read_pose_7_scenes(camera_pose_file);

    const int width  = rgb_img.cols;
    const int height = rgb_img.rows;

    cv::Mat mask;
    cv::Mat world_coordinate = ms_7_scenes_util::camera_depth_to_world_coordinate(camera_depth_img, pose, mask);

    int scaled_width  = width  * scale;
    int scaled_height = height * scale;

    cv::resize(rgb_img, scaled_rgb_img, cv::Size(scaled_width, scaled_height));
    cv::GaussianBlur(scaled_rgb_img, scaled_rgb_img, cv::Size(5, 5), 0.0);

    for (int i = 0; i<num_sample; i++) {
        int x = rand()%scaled_width;
        int y = rand()%scaled_height;

        int x_org = x/scale;
        int y_org = y/scale;

        // ignore bad depth point
        if (mask.at<unsigned char>(y_org, x_org) == 0) {
            continue;
        }
        RGBGLearningSample sp;
        sp.p2d_ = cv::Vec2i(x, y);
        sp.inv_depth_ = 1.0;
        sp.image_index_ = image_index;
        sp.p3d_.x = world_coordinate.at<cv::Vec3d>(y_org, x_org)[0];
        sp.p3d_.y = world_coordinate.at<cv::Vec3d>(y_org, x_org)[1];
        sp.p3d_.z = world_coordinate.at<cv::Vec3d>(y_org, x_org)[2];

        samples.push_back(sp);
    }

    printf("sampled %lu samples\n", samples.size());

    return samples;
}


cv::Point3d RGBGUtil::predictionErrorStddev(const vector<RGBGTestingResult> & results)
{
    assert(results.size() > 0);

    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    for (int i = 0; i<results.size(); i++) {
        cv::Point3d error = results[i].predict_error;
        dx += error.x * error.x;
        dy += error.y * error.y;
        dz += error.z * error.z;
    }

    dx = sqrt(dx/results.size());
    dy = sqrt(dy/results.size());
    dz = sqrt(dz/results.size());

    cv::Point3d stddev(dx, dy, dz);
    return stddev;
}

vector<double> RGBGUtil::predictionErrorDistance(const vector<RGBGTestingResult> & results)
{
    vector<double> distances;

    for (int i = 0; i<results.size(); i++) {
        cv::Point3d error = results[i].predict_error;
        double dis = 0.0;
        dis += error.x * error.x;
        dis += error.y * error.y;
        dis += error.z * error.z;
        dis = sqrt(dis);
        distances.push_back(dis);
    }

    assert(distances.size() == results.size());
    return distances;

}

bool RGBGUtil::readTreeParameter(const char *file_name, RGBGTreeParameter & tree_param)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error: can not open %s\n", file_name);
        return false;
    }

    unordered_map<std::string, int> imap;
    while (1) {
        char s[1024] = {NULL};
        int val = 0;
        int ret = fscanf(pf, "%s %d", s, &val);
        if (ret != 2) {
            break;
        }
        imap[string(s)] = val;
    }
    assert(imap.size() == 11);

    tree_param.is_use_depth_ = (imap[string("is_use_depth")] == 1);
    tree_param.max_frame_num_ = imap[string("max_frame_num")];
    tree_param.sampler_num_per_frame_ = imap[string("sampler_num_per_frame")];

    tree_param.tree_num_ = imap[string("tree_num")];
    tree_param.max_depth_ = imap[string("max_depth")];
    tree_param.min_leaf_node_ = imap[string("min_leaf_node")];

    tree_param.max_pixel_offset_ = imap[string("max_pixel_offset")];
    tree_param.pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
    tree_param.split_candidate_num_ = imap[string("split_candidate_num")];

    tree_param.weight_candidate_num_ = imap[string("weight_candidate_num")];
    tree_param.verbose_ = imap[string("verbose")];

    fclose(pf);
    return true;
}

bool RGBGUtil::readTreePruneParameter(const char *file_name, RGBGTreePruneParameter & param)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error: can not open %s\n", file_name);
        return false;
    }

    unordered_map<std::string, double> imap;
    while (1) {
        char s[1024] = {NULL};
        double val = 0;
        int ret = fscanf(pf, "%s %lf", s, &val);
        if (ret != 2) {
            break;
        }
        imap[string(s)] = val;
    }
    assert(imap.size() == 3);

    param.x_max_stddev_ = imap[string("x_max_stddev")];
    param.y_max_stddev_ = imap[string("y_max_stddev")];
    param.z_max_stddev_ = imap[string("z_max_stddev")];

    fclose(pf);
    return true;
}





