#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cv.hpp>
#include <chrono>
#include "mlinterp.hpp"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


// These are all common classes it's handy to reference with no namespace.
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

using namespace std;
using namespace cv;


//class FeatureExtraction
//    {
//    ros::NodeHandle nh_;
//    image_transport::ImageTransport it_;
//    image_transport::Subscriber image_sub_;
//    image_transport::Publisher feature_pub_;
//
//    public:
//        ImageConverter() : it_(nh_)
//        {
//            // Subscrive to input video feed and publish output video feed
//            image_sub_ = it_.subscribe("/camera/image_raw", 1,
//                                       &ImageConverter::imageCb, this);
//            image_pub_ = it_.advertise("/image_converter/output_video", 1);
//
//            cv::namedWindow(OPENCV_WINDOW);
//        }
//
//        ~ImageConverter()
//        {
//            cv::destroyWindow(OPENCV_WINDOW);
//        }
//}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
    return session_create_status;
    }
    return Status::OK();
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC1, p);
    mat.convertTo(fakeMat, CV_32FC1);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("float32_Cast"), outTensor, tensorflow::DT_FLOAT);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"float32_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return Status::OK();
}

// Driver function to sort the vector elements by
// first element of pair in descending order
bool sortbysecond(const pair<pair<unsigned int, unsigned  int>, float> &a,
                  const pair<pair<unsigned int, unsigned  int>, float> &b)
{
    return (a.second < b.second);
}

// returns row and column indeces and heatmap score where the input array is true
void whereTrue(Eigen::Tensor<bool, 2, 1>& mask, Eigen::Tensor<float, 2, 1>& heatmap, vector<pair<pair<unsigned int, unsigned int>, float>>& pts)
{
    auto dims = mask.dimensions();
    for (int i = 0; i<dims[0]; i++)
    {
        for (int j = 0; j<dims[1]; j++)
        {
            if (mask(i,j))
            {
                pts.push_back(make_pair(make_pair(i, j), heatmap(i,j)));
            }
        }
    }
    // sort by score
    sort(pts.begin(), pts.end(), sortbysecond);
}

// non max supression for pixels
void fastNMS(vector<pair<pair<unsigned int, unsigned int>, float>>& pts, const unsigned int width, const unsigned int height,
             const unsigned int nms_dist)
{
    Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(height + 2 * nms_dist, width + 2 * nms_dist);
    Eigen::MatrixXi ones = Eigen::MatrixXi::Ones(2 * nms_dist + 1, 2 * nms_dist + 1);
    unsigned int i;
    unsigned int j;
    unsigned int deleted = 0;
    unsigned int size = pts.size();
    for(int k = 0; k < size; k++)
    {
        i = pts[k - deleted].first.first;
        j = pts[k - deleted].first.second;
        if(mask(i + nms_dist, j + nms_dist) == 1){
//            cout << " removed point: (" << i << ", " << j << ")\n";
            pts.erase(pts.begin() + k - deleted);
            deleted +=1;
            continue;
        }
        mask(Eigen::seqN(i, 2 * nms_dist + 1), Eigen::seqN(j, 2 * nms_dist + 1)) = ones;
    }
}

// interpolate descriptors
void descInterpolation(vector<pair<pair<unsigned int, unsigned int>, float>>& pts, vector<vector<float>>& descriptors, Eigen::Tensor<float, 4, 1>& coarse_desc)
{
    const unsigned int size = pts.size();
    auto dims = coarse_desc.dimensions();
    // define input dimensions
    unsigned int ni[] = {dims[2], dims[1]};
    // define input points array
    float xi[ni[0]];
    float yi[ni[1]];
    // define input vallue array
    float zi[ni[0] * ni[1]];
    // populate arrays with indeces according to coarse_desc lattice
    for (int i=0; i<ni[0]; ++i) xi[i]=i * 8 + 4;
    for (int i=0; i<ni[1]; ++i) yi[i]=i * 8 + 4;
    // fill in output coordinates
    float xo[size];
    float yo[size];
    for (int i = 0; i < size; ++i)
    {
        xo[i] = pts[i].first.first;
        yo[i] = pts[i].first.second;
    }

    // initialize dscriptor vectors
    for (int i = 0; i < size; ++i)
    {
        vector<float> desc;
        descriptors.push_back(desc);
    }
    // loop over each descriptor dimension
    for (int k = 0; k < dims[3]; ++k) {
        // fill in input values
        for (int i = 0; i < ni[0]; ++i) {
            for (int j = 0; j < ni[1]; ++j) {
                zi[i * ni[1] + j] = coarse_desc(0, j, i, k);
            }
        }
        // Perform the interpolation
        float zo[size]; // Result is stored in this buffer
        mlinterp::interp(
                ni, size,      // Number of points
                zi, zo,        // input and output values
                xi, xo, yi, yo // input and output coordinates
        );
        // populate with interpolated values
        for (int i = 0; i < size; ++i) {
            descriptors[i].push_back(zo[i]);
        }
    }
}

int main(int argc, char* argv[]) {
    // define some constants
    const string root_dir = "";
    const string graph = "src/net_inference/src/half_enc_half_dec.pb";
    const string image_path = "/media/dominic/Extreme SSD/datasets/training_set/-000000-1540457005_400.png";
    const int32 input_width = 640;
    const int32 input_height = 480;
    const string input_layer = "input";
    const vector<string> output_layer = {"heat/BiasAdd", "point_mask/mul", "desc/BiasAdd"};

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
    }

    // read image from harddisk as uint8 image
    Mat image;
    image = imread( image_path, 0);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    // convert image to float32 in range [0,1]
    Mat grayim;
    image.convertTo(grayim, CV_32FC1);
    grayim /= 255;

    // initialize input tensor
    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(input_height);
    shape.AddDim(input_width);
    shape.AddDim(1);
    Tensor tensor;
    tensor = Tensor(tensorflow::DT_FLOAT, shape);
    // read image into tensor
    Status readTensorStatus = readTensorFromMat(grayim, tensor);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return -1;
    }

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, tensor}},{output_layer}, {}, &outputs);
    if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
    }
    // output of network is in row major format
    Eigen::Tensor<float, 4, 1> heat = outputs[0].tensor<float, 4>();
    Eigen::Tensor<bool, 4, 1> mask = outputs[1].tensor<float, 4>().cast<bool>();
    Eigen::Tensor<float, 4, 1> coarse_desc = outputs[2].tensor<float, 4>();

    // softmax along channel dimension
    heat = heat.exp();
    Eigen::array<int, 1> dims({3});
    Eigen::array<int, 4> rep({1, 1, 1, 65});
    Eigen::array<int, 4> four_dim({1, 60, 80, 1});
    Eigen::Tensor<float, 4, 1> t_norm = heat.sum(dims).reshape(four_dim).broadcast(rep);
    heat = heat / t_norm;

    // reshape heatmap into 480x640 tensor after removing dustbin channel
    Eigen::array<int, 4> offset({0, 0, 0, 0});
    Eigen::array<int, 4> extent({1, 60, 80, 64});
    Eigen::Tensor<float, 4, 1> nodust = heat.slice(offset, extent);
    Eigen::array<int, 5> five_dim({1, 60, 80, 8, 8});
    array<int, 5> shuffle_five({0, 1, 3, 2, 4});
    Eigen::array<int, 2> two_dim({480, 640});
    Eigen::Tensor<float, 2, 1> heatmap = nodust.reshape(five_dim).eval().shuffle(shuffle_five).reshape(two_dim);

    // reshape point_mask into 480x640 tensor
    Eigen::Tensor<bool, 2, 1> point_mask = mask.reshape(five_dim).eval().shuffle(shuffle_five).reshape(two_dim);

    // get indeces of feature points and corresponding heatmap score
    vector<pair<pair<unsigned int, unsigned int>, float>> pts;
    whereTrue(point_mask, heatmap, pts);

    // using modified sort() function to sort
    sort(pts.begin(), pts.end(), sortbysecond);

    // nms on thresholded points
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    fastNMS(pts, input_width, input_height, 8);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "NMS time: " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << endl;

    // interpolate descriptors
    begin = chrono::steady_clock::now();
    vector<vector<float>> desc;
    descInterpolation(pts, desc, coarse_desc);
    end = chrono::steady_clock::now();
    cout << "interpolation time: " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << endl;

    // convert to cv2 matrix for display
    heatmap = heatmap.clip(0.001, 1);
    heatmap = heatmap.log();
    Mat heat_gray(input_height, input_width, CV_32F, heatmap.data());
    // normalize heatmap for display
    double min, max;
    minMaxLoc(heat_gray, &min, &max);
    heat_gray = (heat_gray - min) / (max - min) * 255;
    // apply colormap
    Mat tmp;
    heat_gray.convertTo(tmp, CV_8UC1);
    Mat heat_jet;
    applyColorMap(tmp, heat_jet, COLORMAP_JET);
    // overlay image and heatmap
    Mat img_out;
    Mat in[] = {image, image, image};
    merge(in, 3, img_out);
    Mat out_heat;
    addWeighted(heat_jet, 0.7, img_out, 0.3, 0, out_heat);

    // draw circles corresponding to feature locations
    Mat out_circ = img_out.clone();
    for(int i = 0; i < pts.size(); i++)
    {
        Point pt(pts[i].first.second, pts[i].first.first);
        circle(out_circ, pt, 1, Scalar(0, 255, 0), FILLED, LINE_AA);
    }
    Mat out;
    hconcat(out_heat, out_circ, out);
    // show output
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", out);
    waitKey(0);

    return 0;
}
