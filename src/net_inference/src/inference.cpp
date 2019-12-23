#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

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

#include <cv.hpp>

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
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

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

int main(int argc, char* argv[]) {
    string root_dir = "";
    string graph = "src/net_inference/src/half_enc_half_dec.pb";
    string image_path = "/media/dominic/Extreme SSD/datasets/gray_images/eth3d_gray/DSC_0362.png";
    int32 input_width = 640;
    int32 input_height = 480;
    string input_layer = "input";
    vector<string> output_layer = {"heat/BiasAdd", "point_mask/mul", "desc/BiasAdd"};

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
    }

    Mat image;
    image = imread( image_path, 0);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(input_height);
    shape.AddDim(input_width);
    shape.AddDim(1);

    Tensor tensor;
    tensor = Tensor(tensorflow::DT_FLOAT, shape);
    Status readTensorStatus = readTensorFromMat(image, tensor);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return -1;
    }


    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, tensor}},
                                   {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
    }
    Eigen::Tensor<float, 4, 1> heat = outputs[0].tensor<float, 4>();
    heat = heat.exp();
    cout << "heat size: (" << heat.dimensions()[0] << ", " << heat.dimensions()[1] << ", " << heat.dimensions()[2] << ", " << heat.dimensions()[3] << ")\n";

    Eigen::array<int, 1> dims({3});
    Eigen::array<int, 4> rep({1, 1, 1, 65});
    Eigen::array<int, 4> four_dim({1, 60, 80, 1});
    Eigen::Tensor<float, 4, 1> t_norm = heat.sum(dims).eval().reshape(four_dim).eval().broadcast(rep);
    cout << "t_norm size: (" << t_norm.dimensions()[0] << ", " << t_norm.dimensions()[1] << ", " << t_norm.dimensions()[2] << ", " << t_norm.dimensions()[3] << ")\n";

    heat = heat / t_norm;

//    dense = np.reshape(dense, [N, Hc, Wc, 8, 8])
//    dense = np.transpose(dense, [0, 1, 3, 2, 4])
//    heatmap = np.reshape(dense, [N, Hc * 8, Wc * 8])

    Eigen::array<int, 4> offset({0, 0, 0, 0});
    Eigen::array<int, 4> extent({1, 60, 80, 64});
    Eigen::Tensor<float, 4, 1> nodust = heat.slice(offset, extent);

    cout << "nodust size: (" << nodust.dimensions()[0] << ", " << nodust.dimensions()[1] << ", " << nodust.dimensions()[2] << ", " << nodust.dimensions()[3] << ")\n";

    Eigen::array<int, 5> five_dim({1, 60, 80, 8, 8});
    array<int, 5> shuffle_five({0, 1, 3, 2, 4});
    Eigen::array<int, 2> two_dim({480, 640});

    Eigen::Tensor<float, 5, 1> heatmap_5 = nodust.reshape(five_dim).eval().shuffle(shuffle_five);
    cout << "heat_5 size: (" << heatmap_5.dimensions()[0] << ", " << heatmap_5.dimensions()[1] << ", " << heatmap_5.dimensions()[2] << ", " << heatmap_5.dimensions()[3]<< ", " << heatmap_5.dimensions()[4] << ")\n";

    Eigen::Tensor<float, 2, 1> heatmap = heatmap_5.reshape(two_dim);
    cout << "heatmap size: (" << heatmap.dimensions()[0] << ", " << heatmap.dimensions()[1] << ")\n";
    
    Mat out(input_height, input_width, CV_32F, heatmap.data());

//    eigen2cv()

    double min, max;
    minMaxLoc(out, &min, &max);
    // cout << min << endl;
    // cout << max << endl;
    out = (out - min) / (max - min);

//    cout << "M = " << endl << " "  << out << endl << endl;
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", out);
    waitKey(0);

//    ros::init(argc, argv, "super_mario_net_inference");
    return 0;
}
