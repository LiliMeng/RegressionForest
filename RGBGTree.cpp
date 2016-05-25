
//  Authors jimmy and Lili 
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "RGBG_tree.hpp"
#include <iostream>
#include "cvx_io.hpp"
#include <stdio.h>
#include <ctime>

#include <unordered_map>

using std::cout;
using std::endl;


bool RGBGTree::buildTree(const vector<RGBGLearningSample> & samples,
                     const vector<unsigned int> & indices,
                     const vector<cv::Mat> & rgbImages,
                     const RGBGTreeParameter & param)
{
    assert(indices.size() <= samples.size());
    root_ = new RGBGTreeNode(0);

    // set random number
    rng_ = cv::RNG(std::time(0) + 10000);
    param_ = param;
    return this->configureNode(samples, rgbImages, indices, 0, root_);
}

static vector<double> random_number_from_range(double min_val, double max_val, int rnd_num)

{
    assert(rnd_num > 0);

    cv::RNG rng;
    vector<double> data;
    for (int i = 0; i<rnd_num; i++) {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}

static double bestSplitRandomParameter(const vector<RGBGLearningSample> & samples,
                                       const vector<cv::Mat> & rgbImages,
                                       const vector<unsigned int> & indices,
                                       RGBGSplitParameter & split_param,
                                       int min_node_size,
                                       int num_split_random,
                                       vector<unsigned int> & left_indices,
                                       vector<unsigned int> & right_indices)
{
    double min_loss = std::numeric_limits<double>::max();

    ///calculate pixel difference, Feature Response in 2013CVPR paper 2.2
    vector<double> feature_values(indices.size(), 0.0); //0.0 for invalid pixels
    const int c1 = split_param.c1_;
    const int c2 = split_param.c2_;

    for(int i=0; i<indices.size(); i++)
    {
        int index = indices[i];
        assert(index >= 0 && index < samples.size());
        RGBGLearningSample smp = samples[index];
        cv::Point2i p1 = smp.p2d_;
        cv::Point2i p2 = smp.addOffset(split_param.offset2_);
        RGBGLearningSample smp = samples[index];
        cv::Point2i p1 = smp.p2d_;
        cv::Point2i p2 = smp.addOffset(split_param.offset2_);

        const cv::Mat rgb_image = rgbImages[smp.image_index_];

        bool is_inside_image2 = CvxUtil::isInside(rgb_image.cols, rgb_image.rows, p2.x, p2.y);

        double pixel_1_c = 0.0; // out of image as black pixels, random pixel values
        double pixel_2_c = 0.0;

        cv::Vec3b pix_1 = rgb_image.at<cv::Vec3b>(p1.y, p1.x); //(row, col)
        pixel_1_c = pix_1[c1];

        if (is_inside_image2) {
            cv::Vec3b pixel_2 = rgb_image.at<cv::Vec3b>(p2.y, p2.x);
            pixel_2_c = pixel_2[c2];
        }

        feature_values[i] = pixel_1_c * split_param.w1_ + pixel_2_c * split_param.w2_;
    }

    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());

    vector<double> split_values = random_number_from_range(min_v, max_v, num_split_random);  // num_split_random = 20
    // printf("number of randomly selected splitting values is %lu\n", split_values.size());

    ///split data by pixel difference
    bool is_split = false;
    for (int i = 0; i< split_values.size(); i++) {
        double split_v = split_values[i];
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        double cur_loss = 0;
        for (int j =0; j<feature_values.size(); j++)
        {
            int index = indices[j];
            if(feature_values[j] < split_v)
            {
                cur_left_index.push_back(index);
            }
            else
            {
                cur_right_index.push_back(index);
            }
        }
        // if (cur_left_index.size() * 2 < min_node_size || cur_right_index.size() * 2 < min_node_size) {
        //    continue;
        // }

        /*
        cur_loss = RGBGUtil::spatial_variance(samples, cur_left_index);

        if(cur_loss > min_loss) {
            continue;
        }
        cur_loss += RGBGUtil::spatial_variance(samples, cur_right_index);
        */

        if(cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices = cur_left_index;
            right_indices = cur_right_index;
            split_param.threshold_ = split_v;
            //printf("split value is %lf\n", split_param.threshold_);
        }
    }
    if(!is_split)
    {
        return min_loss;
    }
    assert(left_indices.size() + right_indices.size() == indices.size());

    return min_loss;

}


static double bestSplitDecision(const vector<RGBGLearningSample> & samples,
                                const vector<cv::Mat> & rgbImages,
                                const vector<unsigned int> & indices,
                                RGBGSplitParameter & split_param,  // inoutput
                                const vector<cv::Vec2d> & candidate_weights,
                                int min_node_size,
                                int num_split_random,
                                vector<unsigned int> & left_indices,
                                vector<unsigned int> & right_indices)
{
    double min_loss = std::numeric_limits<double>::max();
    RGBGSplitParameter split_param_copy = split_param;
    for (int i = 0; i<candidate_weights.size(); i++) {
        RGBGSplitParameter cur_split_param = split_param_copy;
        cur_split_param.w1_ = candidate_weights[i][0];
        cur_split_param.w2_ = candidate_weights[i][1];
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_loss = bestSplitRandomParameter(samples, rgbImages, indices,
                                                      cur_split_param, min_node_size, num_split_random,
                                                      cur_left_indices, cur_right_indices);
        if (cur_loss < min_loss) {
            min_loss = cur_loss;
            split_param   = cur_split_param;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
        }
    }

    return min_loss;
}

bool RGBGTree::configureNode(const vector<RGBGLearningSample> & samples,
                             const vector<cv::Mat> & rgbImages,
                             const vector<unsigned int> & indices,
                             int depth,
                             RGBGTreeNode *node)
{
    assert(indices.size() < samples.size());

    if(depth >= pram_.max_depth_ || indices.size() <= param_.min_leaf_node_)
    {
        node->depth_ = depth;
        node->is_leaf_ = true;
        node->sample_num_ = (int)indices.size();
        RGBGUtil::mean_stddev(samples, indices, node->p3d_, node->stddev_);
        vector<cv::Vec3d> sample_colors;
        for(int i=0; i<indices.size(); i++)
        {
            sample_colors.push_back(samples[indices[i]].color_);
        }
        RGBGUtil::mean_stddev(sample_colors, node->color_mu_, node->color_sigma_);
        if(param_.verbose_)
        {
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"mean   location: "<<node->p3d_<<endl;
            cout<<"standard  deviation: "<<node->stddev_<<endl;
            cout<<"color mean  :"<<node->color_mu_<<endl;
            cout<<"color stddev :"<<node->color_sigma_<<endl;
        }
        return true;
    }


    const int max_pixel_offset = param_.max_pixel_offset_;
    const int max_channel = 3;
    const int max_random_num   = param_.pixel_offset_candidate_num_;
    const int min_node_size    = param_.min_leaf_node_;
    const int num_split_random = param_.split_candidate_num_;
    const int num_weight = param_.weight_candidate_num_;
    double min_loss = std::numeric_limits<double>::max();

    vector<cv::Vec2d> random_weights;
    random_weights.push_back(cv::Vec2d(1.0, -1.0));

    for (int i = 0; i<num_weight; i++) {
        double w1 = rng_.uniform(-1.0, 1.0);
        double w2 = rng_.uniform(-1.0, 1.0);
        random_weights.push_back(cv::Vec2d(w1, w2));
    }

    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    RGBGSplitParameter split_param;
    bool is_split = false;

    for (int i = 0; i< max_random_num; i++) {
        double x2 = rng_.uniform(-max_pixel_offset, max_pixel_offset);
        double y2 = rng_.uniform(-max_pixel_offset, max_pixel_offset);
        int c1 = rand()%max_channel;
        int c2 = rand()%max_channel;

        RGBGSplitParameter cur_split_param;
        cur_split_param.offset2_ = cv::Point2d(x2, y2);
        cur_split_param.c1_ = c1;
        cur_split_param.c2_ = c2;

        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;

        double cur_loss = bestSplitDecision(samples, rgbImages, indices, cur_split_param,
                                            random_weights,
                                            min_node_size, num_split_random, cur_left_indices, cur_right_indices);

        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = cur_split_param;
        }
    }


    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (param_.verbose_) {
            printf("left, right node number is %lu %lu, percentage: %f loss: %lf\n", left_indices.size(), right_indices.size(), 100.0*left_indices.size()/indices.size(), min_loss);
        }

        node->split_param_ = split_param;
        node->sample_num_ = (int)indices.size();
        //    cout<<"split parameter is "<<split_param.d1_<<" "<<split_param.d2_<<endl;
        if (left_indices.size() != 0) {
            RGBGTreeNode *left_node = new RGBGTreeNode(depth);
            this->configureNode(samples, rgbImages, left_indices, depth + 1, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            RGBGTreeNode *right_node = new RGBGTreeNode(depth);
            this->configureNode(samples, rgbImages, right_indices, depth + 1, right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            node->right_child_ = right_node;
        }
        return true;
    }

    else
    {
        node->depth_ = depth;
        node->is_leaf_ = true;
        node->sample_num_ = (int)indices.size();
        RGBGUtil::mean_stddev(samples, indices, node->p3d_, node->stddev_);
        vector<cv::Vec3d> sample_colors;
        for (int i = 0; i<indices.size(); i++) {
            sample_colors.push_back(samples[indices[i]].color_);
        }
        RGBGUtil::mean_stddev(sample_colors, node->color_mu_, node->color_sigma_ );
        if (param_.verbose_) {
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"mean      location: "<<node->p3d_<<endl;
            cout<<"standard deviation: "<<node->stddev_<<endl;
            cout<<"color mean   : "<<node->color_mu_<<endl;
            cout<<"color stddev : "<<node->color_sigma_<<endl;
        }
        return true;
    }

    return true;
}

bool RGBGTree::pruneLeafNode(RGBGTreeNode * & node,
                             const RGBGTreePruneParameter & param,
                             int & num_pruned_leaf)
{
    assert(node);
    if (node->is_leaf_) {
        cv::Vec3d   stddev = node->stddev_;
        if (stddev[0] > param.x_max_stddev_ ||
            stddev[1] > param.y_max_stddev_ ||
            stddev[2] > param.z_max_stddev_) {
            delete node;
            node = NULL;
            num_pruned_leaf++;
        }
        return true;
    }

    if (node->left_child_) {
        this->pruneLeafNode(node->left_child_, param, num_pruned_leaf);
    }
    if (node->right_child_) {
        this->pruneLeafNode(node->right_child_, param, num_pruned_leaf);
    }
    return true;
}

int RGBGTree::leafNumber(RGBGTreeNode * node) const
{
    if (!node) {
        return 0;
    }
    if (node->is_leaf_) {
        return 1;
    }

    return this->leafNumber(node->left_child_) + this->leafNumber(node->right_child_);
}


bool RGBGTree::pruneTree(const RGBGTreePruneParameter & param)
{
    assert(root_);

    int leaf_num = this->leafNumber(root_);
    int pruned_leaf_num = 0;
    this->pruneLeafNode(root_, param, pruned_leaf_num);

    printf("pruned %d leaf fom %d all leaf, percentage %lf\n", pruned_leaf_num, leaf_num, 1.0*pruned_leaf_num/leaf_num);
    return true;
}

bool RGBGTree::predict(const RGBGLearningSample & sample,
                       const cv::Mat & rgbImage,
                       RGBGTestingResult & predict) const
{
    assert(root_);
    return this->predict(root_, sample, rgbImage, predict);
}

bool RGBGTree::predict(const RGBGTreeNode * const node,
                       const RGBGLearningSample & sample,
                       const cv::Mat & rgbImage,
                       RGBGTestingResult & predict) const
{
    if (node->is_leaf_) {
        predict.predict_p3d_ = node->p3d_;
        predict.predict_error = node->p3d_ - sample.p3d_;        // prediction error
        predict.predict_color_ = node->color_mu_;                // leaf node color
        return true;
    }
    else
    {
        cv::Point2i p1 = sample.p2d_;
        cv::Point2i p2 = sample.addOffset(node->split_param_.offset2_);

        bool is_inside_image2 = CvxUtil::isInside(rgbImage.cols, rgbImage.rows, p2.x, p2.y);
        if (is_inside_image2) {
            cv::Vec3b pixel_1 = rgbImage.at<cv::Vec3b>(p1.y, p1.x);
            cv::Vec3b pixel_2 = rgbImage.at<cv::Vec3b>(p2.y, p2.x);

            double pixel_1_c = pixel_1[node->split_param_.c1_];
            double pixel_2_c = pixel_2[node->split_param_.c2_];
            double split_val = pixel_1_c * node->split_param_.w1_ + pixel_2_c * node->split_param_.w2_;
            if (split_val < node->split_param_.threshold_ && node->left_child_) {
                return this->predict(node->left_child_, sample, rgbImage, predict);
            }
            else if(node->right_child_)
            {
                return this->predict(node->right_child_, sample, rgbImage, predict);
            }
            else
            {
                return false;
            }
        }
        else
        {
            cv::Vec3b pixel_1 = rgbImage.at<cv::Vec3b>(p1.y, p1.x);


            double pixel_1_c = pixel_1[node->split_param_.c1_];
            double pixel_2_c = 0.0;
            double split_val = pixel_1_c * node->split_param_.w1_ + pixel_2_c * node->split_param_.w2_;
            if (split_val < node->split_param_.threshold_ && node->left_child_) {
                return this->predict(node->left_child_, sample, rgbImage, predict);
            }
            else if(node->right_child_)
            {
                return this->predict(node->right_child_, sample, rgbImage, predict);
            }
        }
    }
    return true;
}

// v(S)
double RGBGTree::variance(vector<RGBGLearningSample> labeled_data)
{
    if(labeled_data.size() ==0)
    {
        return 0.0;
    }

    double V = (1.0f /(double)labeled_data.size());
    double sum = 0.0f;

    //calculate mean of S
    cv::Point3f tmp;
    for (auto p : labeled_data)
    {
        tmp.x = tmp.x + p.p3d_.x;
        tmp.y = tmp.y + p.p3d_.y;
        tmp.z = tmp.z + p.p3d_.z;
    }
    uint32_t size = labeled_data.size();
    cv::Point3f mean(tmp.x/size, tmp.y/size, tmp.z/size);

    for (auto p : labeled_data)
    {
        cv::Point3f val;
        val.x = p.p3d_.x - p.p3d_.x;
        val.y = p.p3d_.y - p.p3d_.y;
        val.z = p.p3d_.z - p.p3d_.z;

        sum += val.x*val.x + val.y*val.y +val.z*val.z;

    }

    return V*sum;
}


 //Q(S_n, \theta)
double RGBGTree::objective_function(vector<RGBGLearningSample> data, vector<RGBGLearningSample> left, vector<RGBGLearningSample> right)
{
        double var = variance(data);
        double left_var = ((double)left.size()/(double)data.size())*variance(left);
        double right_var = ((double)right.size()/(double)data.size())*variance(right);

        return var-(left_var + right_var);
}

Eigen::Vector3d RGBGTree::GetLeafMode(std::vector<RGBGLearningSample> S)
{
    std::vector<Eigen::Vector3d> data;

    //calculate mode for leaf,sub_sample N_SS = 500
    for (uint16_t i=0; i<(S.size()<500 ? S.size(): 500); i++)
    {
        auto p = S[i];
        Eigen::Vector3d point {p.p3d_.x, p.p3d_.y, p.p3d_.z};
        data.push_back(point);
    }

    //cluster
    MeanShift ms = MeanShift(nullptr);
    double kernel_bandwidth = 0.01f; //gaussian
    std::vector<Eigen::Vector3d> cluster = ms.cluster(data, kernel_bandwidth);

    // find mode
    std::vector<Point3D> clustered_points;
    for(auto c : cluster)
    {

        clustered_points.push_back(Point3D(floor(c[0]*10000)/10000,
                                           floor(c[1]*10000)/10000,
                                           floor(c[2]*10000)/10000));
    }

    Point3DMap cluster_map;

    for(auto p : clustered_points)
    {
        cluster_map[p]++;
    }

    std::pair<Point3D, uint32_t> mode(Point3D(0.0, 0.0, 0.0), 0);

    for(auto p : cluster_map)
    {
        if(p.second > mode.second)
        {
            mode = p;
        }
    }

    return Eigen::Vector3d(mode.first.x, mode.first.y, mode.first.z);

}

DECISION eval_learner()
{
    bool valid = true;
    float response =

}

/*************************      RGBGTreeNode      *******************/
void write_RGBG_prediction(FILE *pf, RGBGTreeNode * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }

    //write current node
    RGBGSplitParameter param = node->split_param_;
    fprintf(pf, "%2d %d %6d %3.2f %d\t %12.6f %d\t %12.6f %12.6f %12.6f\t",
            node->depth_, (int)node->is_leaf_, node->sample_num_, node->sample_percentage_, param.c1_,
            param.offset2_.x, param.offset2_.y, param.c2_,
            param.w1_, param.w2_, param.threshold_);

    fprintf(pf, "%6.3f %6.3f %6.3f\t %6.3f %6.3f %6.3f\t %6.1f %6.1f %6.1f\t %6.1f %6.1f %6.1f\n",
            node->p3d_.x, node->p3d_.y, node->p3d_.z,
            node->stddev_[0], node->stddev_[1], node->stddev_[2],
            node->color_mu_[0], node->color_mu_[1], node->color_mu_[2],
            node->color_sigma_[0], node->color_sigma_[1], node->color_sigma_[2]);

    write_RGBG_prediction(pf, node->left_child_);
    write_RGBG_prediction(pf, node->right_child_);
}


bool RGBGTreeNode::writeTree(const char *fileName, RGBGTreeNode * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("cannot open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t sample_num\t sample_percentage\t c1\t displace2\t c2\t w1 w2\t threshold\t wld_3d stddev \t mean_color stddev_color\n");
    write_RGBG_prediction(pf, root);
    fclose(pf);
    return true;
}

static void read_RGBG_prediction(FILE *pf, RGBGTreeNode * & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if(!ret) {
        node = NULL;
        return;
    }
    if(lineBuf[0] == '#') {
        //empty node
        node = NULL;
        return;
    }

    // read node parameters
    node = new RGBGTreeNode();
    assert(node);
    int depth = 0;
    int isLeaf = 0;
    int sample_num = 0;
    double sample_percentage = 0.0;

    double d2[2] = {0.0};
    int c1 = 0;
    int c2 = 0;
    double wt[2] = {0.0};
    double threshold = 0;
    double xyz[3] = {0.0};
    double sigma_xyz[3] = {0.0};
    double color[3] = {0.0};
    double color_sigma[3] = {0.0};


    int ret_num = sscanf(lineBuf, "%d %d %d %lf %d %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                         &depth, &isLeaf, & sample_num, &sample_percentage,
                         &c1,
                         &d2[0], &d2[1], &c2,
                         &wt[0], &wt[1],
                         &threshold,
                         &xyz[0], &xyz[1], &xyz[2],
                         &sigma_xyz[0], &sigma_xyz[1], &sigma_xyz[2],
                         &color[0], &color[1], &color[2],
                         &color_sigma[0], &color_sigma[1], &color_sigma[2]);

    assert(ret_num == 23);

    node->depth_ = depth;
    node->is_leaf_ = (isLeaf ==1);
    node->p3d_ = cv::Point3d(xyz[0], xyz[1], xyz[2]);
    node->stddev_ = cv::Vec3d(sigma_xyz[0], sigma_xyz[1], sigma_xyz[2]);
    node->color_mu_ = cv::Vec3d(color);
    node->color_sigma_ = cv::Vec3d(color_sigma);
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;


    RGBGSplitParameter param;
    param.offset2_ = cv::Point2d(d2[0], d2[1]);
    param.c1_ = c1;
    param.c2_ = c2;
    param.w1_ = wt[0];
    param.w2_ = wt[1];
    param.threshold_ = threshold;

    node->split_param_ = param;

    node->left_child_ = NULL;
    node->right_child_ = NULL;

    read_RGBG_prediction(pf, node->left_child_);
    read_RGBG_prediction(pf, node->right_child_);
}

bool RGBGTreeNode::readTree(const char *fileName, RGBGTreeNode * & root)
{
    FILE *pf = fopen(fileName, "r");
    if(!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }

    // read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf),pf);
    printf("%s\n", line_buf);
    read_RGBG_prediction(pf, root);
    fclose(pf);
    return true;
}

