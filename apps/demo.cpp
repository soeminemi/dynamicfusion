#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
using namespace kfusion;

struct DynamicFusionApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        DynamicFusionApp& kinfu = *static_cast<DynamicFusionApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.show_warp(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.interactive_mode_ = !kinfu.interactive_mode_;
    }

    DynamicFusionApp(std::string dir) : exit_ (false), interactive_mode_(false), pause_(false), directory(true), dir_name(dir)
    {
        KinFuParams params = KinFuParams::default_params_dynamicfusion();
        kinfu_ = KinFu::Ptr( new KinFu(params) );
        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);

    }
    static void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
        cv::waitKey(10);
    }

    void show_raycasted(KinFu& kinfu, int i)
    {
        const int mode = 0;
        if (interactive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);

#ifdef OUTPUT_PATH
        std::string path = TOSTRING(OUTPUT_PATH) + std::to_string(i) + ".jpg";
        cv::imwrite(path, view_host_);
#endif
        // cv::warpAffine(src, dst, rot_mat, cv::Size(720,1280));
        cv::transpose(view_host_, dst);
        cv::imshow("Scene", dst);
        cv::waitKey(10);

    }

    void show_warp(KinFu &kinfu)
    {
        cv::Mat warp_host =  kinfu.getWarp().getNodesAsMat();
        viz.showWidget("warp_field", cv::viz::WCloud(warp_host));
    }

    bool execute()
    {
        video.open("output1.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(720, 1280));
        if(!video.isOpened())
        {
            std::cout<<"open video writer failed"<<std::endl;
        }
        bool flag_show = true;
        KinFu& dynamic_fusion = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        std::vector<cv::String> depths;             // store paths,
        std::vector<cv::String> images;             // store paths,

        cv::glob(dir_name + "/depth", depths);
        cv::glob(dir_name + "/color", images);

        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());

        for (int i = 300; i < depths.size() && !exit_ && !viz.wasStopped(); i++) {
            image = cv::imread(images[i], cv::IMREAD_COLOR);
            if(i > 580)
                continue;
            depth = cv::imread(depths[i], cv::IMREAD_ANYDEPTH);
            depth = depth / 4;
            for (size_t i = 0; i < depth.rows; i++)
            {
                for (size_t j = 0; j < depth.cols; j++)
                {
                    if(depth.at<ushort>(i,j)>1500)
                    {
                        depth.at<ushort>(i,j) = 0;
                    }
                }
                
            }
            
            std::cout<<"upload depth data"<<std::endl;
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

//            {
//                SampledScopeTime fps(time_ms);
//                (void) fps;
            std::cout<<"start fusion"<<std::endl;
            has_image = dynamic_fusion(depth_device_);
//            }
            std::cout<<"start to show result: "<<flag_show<<std::endl;
            if (has_image && flag_show){
                show_raycasted(dynamic_fusion, i);
                std::cout<<"write to the video: "<<dst.rows<<", "<<dst.cols<<std::endl;
                video<<(dst);
                std::stringstream sss;
                sss<<"./videos/frame_"<<i<<".jpg";
                cv::imwrite(sss.str(),dst);
            }
            if(flag_show)
            {
                show_depth(depth);
                cv::imshow("Image", image);

                if (!interactive_mode_) {
                    viz.setViewerPose(dynamic_fusion.getCameraPose());
                }

                int key = cv::waitKey(pause_ ? 0 : 3);
                show_warp(dynamic_fusion);
                switch (key) {
                    case 't':
                    case 'T' :
                        show_warp(dynamic_fusion);
                        break;
                    case 'i':
                    case 'I' :
                        interactive_mode_ = !interactive_mode_;
                        break;
                    case 27:
                        exit_ = true;
                        break;
                    case 32:
                        pause_ = !pause_;
                        break;
                }
                viz.spinOnce(3, true);
                //exit_ = exit_ || i > 100;
            }
        }

        video.release();
        std::cout<<"release the video writer"<<std::endl;
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, interactive_mode_, directory;
    std::string dir_name;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cv::VideoWriter video;
    cv::Mat dst;

};

int main (int argc, char* argv[])
{
    assert(argc == 2 && "Usage: ./dynamicfusion <data-directory>");
    std::cout<<"start the project with path: "<<argv[1]<<std::endl;
    DynamicFusionApp *app;
    app = new DynamicFusionApp(argv[1]);
    std::cout<<"start excute all thread"<<std::endl;
    // executing
    try { app->execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    delete app;
    return 0;
}
