#ifndef CONFIG_H
#define CONFIG_H
#include "include.h"
#endif

#define ENABLE_LOG 1 // ��־���
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

// Ԥ��ģʽ�����ǰ�����ֱ��ʵ�Ϊ0.6���ˣ�
bool preview = false;
// GPU����
bool try_cuda = false;
// ����ƥ��ֱ��ʣ�0.6 = ��С��0.6�������أ�����-1��
double work_megapix = 0.6;
// ����߹��Ʒֱ���
double seam_megapix = 0.1;
// ����ƴ��ͼ��ֱ���
double compose_megapix = -1;
// ���α任���Ŷȣ�ɾ���������Ŷȵ�ͼƬ��
float conf_thresh = 0;
// ����ƥ������Ŷ�
#ifdef HAVE_OPENCV_XFEATURES2D
// ���������ࣨsurf��orb��sift��akaze��
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
// ƥ���㷨��͸�ӣ�homography�����䣺affine��
string matcher_type = "affine";
// ͼ��ƥ�䷶Χ��-1��ȫ���ӣ�������ȫ����ֻ��homography��
int range_width = -1;
// ���α任������
string estimator_type = "homography";
// BA���ۺ���
//����С����ͶӰ��reproj����С��������ray����С��������affine����������no��
string ba_cost_func = "ray";
// BA�Ż�������<fx><skew><ppx><aspect><ppy>��
// x��ʾ�Ż���_��ʾ���Ż�
string ba_refine_mask = "x_x_x";
// ����ЧӦ������ˮƽ����WAVE_CORRECT_HORIZ����ֱ����WAVE_CORRECT_VERT��
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_VERT;
// �Ƿ񱣴�ƥ���ϵͼ
bool save_graph = false;
std::string save_graph_to; // ����·��
string result_name = "C:/Users/Administrator/Desktop/stitch_code/data1/result.jpg"; // �ļ�����
// ͼ��ͶӰ��������
// affine ����ͶӰ���ʺϼ�ƽ�泡��
// plane ƽ��ͶӰ��������С�ӽ�ƴ��
// cylindrical Բ��ͶӰ���ʺ�ˮƽ�ӽǽϴ��ȫ��ͼ
// spherical ����ͶӰ��Ĭ�ϣ����ʺϴ��ӽ�ȫ��ͼ
// fisheye ����ͶӰ��ģ�����۾�ͷЧ��
// stereographic ����ͶӰ���ʺ�������ʧ������ȫ��ͼ
// compressedPlaneA2B1 ѹ��ƽ��ͶӰ�����ٻ���
// compressedPlaneA1.5B1 ѹ��ƽ��ͶӰ���ʺ��е��ӽ�
// compressedPlanePortraitA2B1 ѹ��ƽ������ͶӰ���ʺ���ֱ����ƴ��
// compressedPlanePortraitA1.5B1 ѹ��ƽ������ͶӰ���ʺ��е��ӽ�
// paniniA2B1 Panini ͶӰ������ѧ����ͼЧ��
// paniniA1.5B1 Panini ͶӰ���ʺ��е��ӽǽ�������
// paniniPortraitA2B1 Panini ����ͶӰ���ʺ���ֱ����ƴ��
// paniniPortraitA1.5B1 Panini ����ͶӰ���ʺ��е���ֱ�ӽ�
// mercator ī����ͶӰ���ʺϵ�ͼ����ӽǳ���
// transverseMercator ����ī����ͶӰ�������ڼ������������ӽ�ƴ��
string warp_type = "compressedPlaneA2B1";
// ������㷨
//��ͼ������voronoi��������ɫ��ͼ�и����gc_color��������ɫ���ݶȵ�ͼ�и����gc_colorgrad��
//��������ɫ�Ķ�̬�滮��dp_color��������ɫ���ݶȵĶ�̬�滮��dp_colorgrad��
string seam_find_type = "gc_color";
// �عⲹ��
//���ֿ����油����GAIN_BLOCKS��ȫ�����油����GAIN����������no��
//����RGBȫ�����油����CHANNELS����RGB�ֿ鲹����CHANNELS_BLOCKS��
int expos_comp_type = ExposureCompensator::GAIN;
int expos_comp_nr_feeds = 1; // ������������
int expos_comp_nr_filtering = 2; // ������˵�������
int expos_comp_block_size = 32; // �ֿ��С
// ͼ���ںϣ�����ںϣ�MULTI_BAND�����ںϣ�FEATHER�����ںϣ�no��
int blend_type = Blender::FEATHER;
// �ں�ǿ�ȣ���Χ[0,100]��
float blend_strength = 5;
// ʱ�����������ԭͼ��ʽ�����AS_IS���ü���ʽ�����CROP��
bool timelapse = false;
int timelapse_type = Timelapser::AS_IS;

/*
���䣺
1.����ЧӦ�������ת��ͷ������ɵ� ͼ��߽�Ť������״����
2.����߷�����ͼ������죬ɫ���ĵط�ƴ������
3.ͼ���ںϷ������𻯣���ԵȨ�ظ�С
4.�ں�ǿ�ȣ�ֵԽ�ߣ��ں�Խƽ����������ģ������֮���ܳ���ƴ�Ӻۼ�
5.ʱ��������������δƴ�ӵ�����ͼƬ
*/



int stitch()
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount(); 
#endif


    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("����ͼƬ��������2��");
        return -1;
    }

    // ������ȡ����������߹��Ʊ��������ͼ�����
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("======= ��������ȡ�� ======="); 
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
        cout << "����������δע��: '" << features_type << "'.\n";
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
            LOGLN("�����޷���ͼƬ " << img_names[i]);
            return -1;
        }
        // ͼƬ�ֱ��ʵ���
        if (work_megapix < 0) // δ�����򱣳�ԭ�ֱ���
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
        LOGLN("ͼ #" << i + 1 << " ����������: " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("��ȡ�������ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= ������ƥ���� =======");
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

    (*matcher)(features, pairwise_matches); // ���ߴ����ƥ���������ڵ����������Ŷȷ���
    matcher->collectGarbage(); // �ͷ���Դ

    LOGLN("ƥ���������ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    if (save_graph)
    {
        LOGLN("����ƥ��ͼƬ��...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // ɸѡ�������Ŷȵ�ͼƬ
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

    //std::cout << "======= ��鼸�ι������� =======" << std::endl;
    //std::cout << "������������" << features.size() << std::endl;
    //for (size_t i = 0; i < features.size(); ++i) {
    //    std::cout << "ͼ #" << i + 1 << " ����������: " << features[i].keypoints.size() << std::endl;
    //}

    //std::cout << "ƥ���������" << pairwise_matches.size() << std::endl;
    //for (size_t i = 0; i < pairwise_matches.size(); ++i) {
    //    std::cout << "ƥ��� #" << i + 1
    //        << " �ڵ�����: " << pairwise_matches[i].num_inliers
    //        << ", ���Ŷ�: " << pairwise_matches[i].confidence << std::endl;
    //}

    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("����ɸѡ��ͼƬ����2��");
        return -1;
    }
    // cout << "���Ŷ�ɸѡ��ͼƬ������ " << num_images << endl;

    // ���ι���
    LOGLN("======= �������� =======");
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
        cout << "���󣺼��ι���ʧ�� \n";
        return -1;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        // LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    // ������
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "������������֧��: '" << ba_cost_func << "' \n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh); // ������ֵ
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U); // �Ż�����
    if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1; // ���� fx
    if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1; // ����ƫ�� skew
    if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1; // ���ĺ����� ppx
    if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1; // ������ aspect
    if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1; // ���������� ppy
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "����������ʧ�� \n";
        return -1;
    }
    LOGLN("��������ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // �����������
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

    // ����ЧӦ����
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("======= ͼ��任�� =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // ��ʼ������
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
        cout << "�����޷�����ͶӰ '" << warp_type << "'\n";
        return 1;
    }
    // ���α任
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

    LOGLN("ͼ��任��ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= �عⲹ���� =======");
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

    LOGLN("�عⲹ����ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("======= Ѱ�ҷ������ =======");
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
        cout << "�����޷���������߲����� '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("����߲��Һ�ʱ�� " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("======= ͼƬƴ���� =======");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender; // �ں���
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("ƴ��ͼƬ #" << indices[img_idx] + 1);

        // ����ͼ���С
        full_img = imread(samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            compose_work_aspect = compose_scale / work_scale;

            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // ����ͶӰ����
            for (int i = 0; i < num_images; ++i)
            {
                // �����ڲ�
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // ����ͼ���С
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                // ����ͶӰ����
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

        // ͶӰ�任
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // �任ͼ������
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // �عⲹ��
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

        LOGLN("ƴ�Ӻ�ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        imwrite(result_name, result);
    }

    LOGLN("�ܺ�ʱ: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" <<endl);

    // �洢ƴ�Ӻõ�ͼƬ
    img_names.clear();
    img_names.push_back(result_name);

    return 0;
}