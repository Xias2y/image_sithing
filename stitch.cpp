#ifndef CONFIG_H
#define CONFIG_H
#include "include.h"
#endif

#define ENABLE_LOG 1 // 日志输出
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

// 预览模式（就是把输出分辨率调为0.6罢了）
bool preview = false;
// GPU加速
bool try_cuda = false;
// 特征匹配分辨率（0.6 = 缩小到0.6百万像素）（可-1）
double work_megapix = 0.4;
// 缝合线估计分辨率
double seam_megapix = 0.1;
// 最终拼接图像分辨率
double compose_megapix = -1;
// 几何变换置信度（删除低于置信度的图片）
float conf_thresh = 1.f;
// 特征匹配点置信度
#ifdef HAVE_OPENCV_XFEATURES2D
// 特征点种类（surf、orb、sift、akaze）
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
// 匹配算法（透视：homography、仿射：affine）
string matcher_type = "homography";
// 图像匹配范围（-1：全连接）（不想全连接只能homography）
int range_width = -1;
// 几何变换估计器
string estimator_type = "homography";
// BA代价函数
//（最小化重投影误差：reproj、最小化光线误差：ray、最小化仿射误差：affine、不调整：no）
string ba_cost_func = "ray";
// BA优化参数（<fx><skew><ppx><aspect><ppy>）
// x表示优化，_表示不优化
string ba_refine_mask = "x___x";
// 波形效应矫正（水平方向：WAVE_CORRECT_HORIZ、垂直方向：WAVE_CORRECT_VERT）
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_VERT;
// 是否保存匹配关系图
bool save_graph = false;
std::string save_graph_to; // 保存路径

// string result_name = "C:/Users/Administrator/Desktop/stitch_code/result.png"; // 文件名称

// 图像投影几何类型
// affine 仿射投影，适合简单平面场景
// plane 平面投影，适用于小视角拼接
// cylindrical 圆柱投影，适合水平视角较大的全景图
// spherical 球面投影（默认），适合大视角全景图
// fisheye 鱼眼投影，模拟鱼眼镜头效果
// stereographic 立体投影，适合生成无失真球形全景图
// compressedPlaneA2B1 压缩平面投影，减少畸变
// compressedPlaneA1.5B1 压缩平面投影，适合中等视角
// compressedPlanePortraitA2B1 压缩平面人像投影，适合竖直方向拼接
// compressedPlanePortraitA1.5B1 压缩平面人像投影，适合中等视角
// paniniA2B1 Panini 投影，建筑学正视图效果
// paniniA1.5B1 Panini 投影，适合中等视角建筑场景
// paniniPortraitA2B1 Panini 人像投影，适合竖直方向拼接
// paniniPortraitA1.5B1 Panini 人像投影，适合中等竖直视角
// mercator 墨卡托投影，适合地图或宽视角场景
// transverseMercator 横向墨卡托投影，适用于极地区域或横向视角拼接
string warp_type = "plane";
// 缝合线算法
//（图方法：voronoi、基于颜色的图切割方法：gc_color、基于颜色和梯度的图切割方法：gc_colorgrad）
//（基于颜色的动态规划：dp_color、基于颜色和梯度的动态规划：dp_colorgrad）
string seam_find_type = "gc_colorgrad";
// 曝光补偿
//（分块增益补偿：GAIN_BLOCKS、全局增益补偿：GAIN、不补偿：no）
//（按RGB全局增益补偿：CHANNELS、按RGB分块补偿：CHANNELS_BLOCKS）
int expos_comp_type = ExposureCompensator::NO;
int expos_comp_nr_feeds = 1; // 补偿迭代次数
int expos_comp_nr_filtering = 2; // 增益过滤迭代次数
int expos_comp_block_size = 32; // 分块大小
// 图像融合（多带融合：MULTI_BAND、羽化融合：FEATHER、不融合：no）
int blend_type = Blender::MULTI_BAND;
// 融合强度（范围[0,100]）
float blend_strength = 15;
// 时间序列输出（原图方式输出：AS_IS、裁剪方式输出：CROP）
bool timelapse = false;
int timelapse_type = Timelapser::AS_IS;

/*
补充：
1.波形效应：相机旋转或镜头畸变造成的 图像边界扭曲或波浪状变形
2.缝合线方法：图方法最快，色差大的地方拼接明显
3.图像融合方法：羽化：边缘权重更小
4.融合强度：值越高，融合越平滑，但可能模糊；反之可能出现拼接痕迹
5.时间序列输出：输出未拼接的连续图片
*/


int tmp = 0; // 拼接结果编号

int stitch()
{
    tmp++;
#if ENABLE_LOG
    int64 app_start_time = getTickCount(); 
#endif

    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("错误：图片输入少于2张");
        return -1;
    }

    // 特征提取比例、缝合线估计比例、输出图像比例
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("======= 特征点提取中 ======="); 
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else if (features_type == "akaze")
    {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else
    {
        cout << "错误：特征点未注册: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();

        // cout << img_names[i] << endl;

        if (full_img.empty())
        {
            LOGLN("错误：无法打开图片 " << img_names[i]);
            return -1;
        }
        // 图片分辨率调整
        if (work_megapix < 0) // 未设置则保持原分辨率
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        LOGLN("图 #" << i + 1 << " 特征点数量: " << features[i].keypoints.size() << "  " << img_names[i]);

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("提取特征点耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= 特征点匹配中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width == -1)
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    (*matcher)(features, pairwise_matches); // 后者存放了匹配点对数，内点数量，置信度分数
    matcher->collectGarbage(); // 释放资源

    LOGLN("匹配特征点耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    if (save_graph)
    {
        LOGLN("保存匹配图片中...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // 筛选满足置信度的图片
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    //std::cout << "======= 检查几何估计输入 =======" << std::endl;
    //std::cout << "匹配对数量：" << pairwise_matches.size() << std::endl;
    //for (size_t i = 0; i < pairwise_matches.size(); ++i) {
    //    std::cout << "匹配对 #" << i + 1
    //        << " 内点数量: " << pairwise_matches[i].num_inliers
    //        << ", 置信度: " << pairwise_matches[i].confidence << std::endl;
    //}

    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("错误：筛选后图片少于2张");
        return -1;
    }
    cout << "置信度筛选后图片数量： " << num_images << endl;

    // 几何估计
    LOGLN("======= 束调整中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif
    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "错误：几何估计失败 \n";
        return -1;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        // LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    // 束调整
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "束调整方法不支持: '" << ba_cost_func << "' \n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh); // 设置阈值
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U); // 优化掩码
    if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1; // 焦距 fx
    if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1; // 像素偏移 skew
    if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1; // 光心横坐标 ppx
    if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1; // 长宽比 aspect
    if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1; // 光心纵坐标 ppy
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "错误：束调整失败 \n";
        return -1;
    }
    LOGLN("束调整耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // 估计相机焦距
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        // LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    // 波形效应矫正
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("======= 图像变换中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // 初始化掩码
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    Ptr<WarperCreator> warper_creator;

#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }
    if (!warper_creator)
    {
        cout << "错误：无法创建投影 '" << warp_type << "'\n";
        return 1;
    }
    // 几何变换
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("图像变换耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= 曝光补偿中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    LOGLN("曝光补偿耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= 寻找缝合线中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "错误：无法创建缝合线查找器 '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("缝合线查找耗时： " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("======= 图片拼接中 =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender; // 融合器
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("拼接图片 #" << indices[img_idx] + 1);

        // 重置图像大小
        full_img = imread(samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            compose_work_aspect = compose_scale / work_scale;

            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // 更新投影区域
            for (int i = 0; i < num_images; ++i)
            {
                // 更新内参
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // 更新图像大小
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                // 计算投影区域
                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // 投影变换
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // 变换图像掩码
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // 曝光补偿
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                // LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f / blend_width);
                // LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        if (timelapse)
        {
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }
    
    if (!timelapse)
    {
        Mat result, result_mask;
        blender->blend(result, result_mask);

        LOGLN("拼接耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        string base_path = "C:/Users/Administrator/Desktop/images/output";
        string result_name = base_path + std::to_string(tmp) + ".jpg";
        
        imwrite(result_name, result);

    }

    LOGLN("总耗时: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    return 0;
}