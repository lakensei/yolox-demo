//package com.example.yolox;
//
//import lombok.extern.slf4j.Slf4j;
//import org.bytedeco.javacpp.DoublePointer;
//import org.junit.Test;
//
//import org.bytedeco.javacpp.FloatPointer;
//import org.bytedeco.javacpp.IntPointer;
//import org.bytedeco.javacpp.indexer.FloatIndexer;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_dnn.Net;
//import org.bytedeco.opencv.opencv_text.FloatVector;
//import org.bytedeco.opencv.opencv_text.IntVector;
//
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.List;
//
//import static org.bytedeco.opencv.global.opencv_core.*;
//import static org.bytedeco.opencv.global.opencv_dnn.*;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
//
///**
// * @author : 互联网办外网01
// * @date : 2022-05-17 16:43
// */
//@Slf4j
//public class YoloxTest {
//
//    @Test
//    public void main() {
//        String rootPath = "E:/luo/yolo-demo/data";
//        // 结果图片路径
//        String outImgPath = rootPath + "/images/res.jpg";
//        // 读取模型
//        Net net = readNet(rootPath + "/model/yolox_s.onnx");
//        // 待预测图片
//        Mat src = imread(rootPath + "/images/dog.jpg");
//        // 分类名称
//        List<String> names = null;
//        try {
//            names = Files.readAllLines(Paths.get(rootPath + "/model/coco.names"));
//        } catch (IOException e) {
//            log.info("获取分类名称失败，文件路径[]");
//        }
//        assert names != null;
//
//        // ============修改图片尺寸开始============
//        int width = 640;
//        int height = 640;
//        double scale = Math.min(width / (src.cols() * 1.0), height / (src.rows() * 1.0));
//        int upW = (int) (scale * src.cols());
//        int upH = (int) (scale * src.rows());
//
//        Mat dstimg = new Mat(upH, upW, CV_8UC3);
//        Mat outimg = new Mat(height, width, CV_8UC3, new Scalar(144, 144, 144, 0));
//        resize(src, dstimg, dstimg.size());
//        /*
//        https://blog.csdn.net/sandalphon4869/article/details/94565827
//        https://github.com/bytedeco/javacv/issues/795
//        https://github.com/bytedeco/javacv/issues/475
//         */
//        Mat roi = outimg.apply(new Rect(0,0,upW, upH));
//        dstimg.copyTo(roi);
////        dstimg.copyTo(outimg.apply(new Range(0, upW), new Range(0, upH)));  // 上面new Mat 宽高反了的时候 此处有效果 上面roi区域改变，原图并没有改变
////        imwrite("E:\\luo\\yolo-demo\\data\\yolox-opencv-dnn-main/outimg.jpg", outimg);
//        // ============修改图片尺寸结束============
//
//        // ============归一化开始============
//        // 需根据模型判断 预处理需不需要 "BGR2RGB，除以255.0, 减均值除以方差" 这几步
//
//        // 颜色模式转换 BGR TO RGB
//        cvtColor(outimg, outimg, COLOR_BGR2RGB);
//        outimg.convertTo(outimg, CV_32F);
//
//        double[] mean = {0.485, 0.456, 0.406};
//        double[] std = {0.229, 0.224, 0.225};
//        FloatIndexer srcIndexer = outimg.createIndexer();
//        for (int x = 0; x < srcIndexer.rows(); x++) {
//            for (int y = 0; y < srcIndexer.cols(); y++) {
//                float[] values = new float[3];
//                srcIndexer.get(x, y, values);
//                values[0] = (float) ((values[0] / 255.0 - mean[0]) / std[0]);
//                values[1] = (float) ((values[1] / 255.0 - mean[1]) / std[1]);
//                values[2] = (float) ((values[2] / 255.0 - mean[2]) / std[2]);
//                srcIndexer.put(x, y, values);
//            }
//        }
//
//        // ============归一化结束============
//
//        // 输入模型进行预测
//        StringVector outNames = net.getUnconnectedOutLayersNames();
//        Mat blob = blobFromImage(outimg);
//
//        net.setInput(blob);
//        MatVector outs = new MatVector(outNames.size());
//        net.forward(outs, outNames);
//        // 释放资源
//        blob.release();
//
//
//        // 处理模型结果
//        final IntVector classIds = new IntVector();
//        final FloatVector confidences = new FloatVector();
//        final RectVector boxes = new RectVector();
//
//        Mat outs0 = outs.get(0);
//
//        if (outs0.dims() == 3) {
//            int numProposal = outs0.size(1);
//            outs0 = outs0.reshape(0, numProposal);
//        }
//
//        int rowInd = 0;
//
////        int nout = names.size() + 5;
//        int[] stride = { 8, 16, 32 };
//        FloatIndexer pdata = outs0.createIndexer();
//        float probThreshold = 0.5f;
//        float nmsThreshold = 0.5f;
//        double e = Math.E;
//
//        for(int n=0; n< 3; n++) {
//            int numGridX = width / stride[n];
//            int numGridY = height / stride[n];
//            for (int i = 0; i < numGridX; i++)
//            {
//                for (int j = 0; j < numGridY; j++) {
//
//                    float boxScore = pdata.get(rowInd,4 );
//
//                    Mat scores = outs0.row(rowInd).colRange(5, outs0.cols());
//
//                    DoublePointer maxVal= new DoublePointer();
//                    Point max = new Point();
//                    // 找出图像中最大值最小值
//                    minMaxLoc(scores, null, maxVal, null, max, null);
//
//                    int classIdx = max.x();
//                    float clsScore = pdata.get(rowInd, 5 + classIdx);
//                    float boxProb = boxScore * clsScore;
//                    if (boxProb > probThreshold)
//                    {
////                        log.info("maxScore {}, classId{}", boxProb, classIdx);
//                        double xCenter = (pdata.get(rowInd, 0) + j) * stride[n];
//                        double yCenter = (pdata.get(rowInd,1) + i) * stride[n];
//                        double w = Math.pow(e, pdata.get(rowInd,2)) * stride[n];
//                        double h = Math.pow(e, pdata.get(rowInd,3)) * stride[n];
//                        double x0 = xCenter - w * 0.5f;
//                        double y0 = yCenter - h * 0.5f;
//
//                        classIds.push_back(classIdx);
//                        confidences.push_back(boxProb);
//                        boxes.push_back(new Rect((int)x0, (int) y0, (int)(w), (int)(h)));
//                    }
//                    rowInd++;
//                }
//            }
//        }
//        // 资源释放
//        pdata.release();
//        outs0.release();
//
//        //  图片加上预测结果
//        IntPointer indices = new IntPointer(confidences.size());
//        FloatPointer confidencesPointer = new FloatPointer(confidences.size());
//        confidencesPointer.put(confidences.get());
//
//        // 非极大值抑制
//        NMSBoxes(boxes, confidencesPointer, probThreshold, nmsThreshold, indices, 1.f, 0);
//
//        for (int i = 0; i < indices.limit(); ++i) {
//            final int idx = indices.get(i);
//            final Rect box = boxes.get(idx);
//
//            double x0 = (box.x()) / scale;
//            double y0 = (box.y()) / scale;
//            double x1 = (box.x() + box.width()) / scale;
//            double y1 = (box.y() + box.height()) / scale;
//
//            // clip
//            x0 = Math.max(Math.min(x0, (float)(src.cols() - 1)), 0.f);
//            y0 = Math.max(Math.min(y0, (float)(src.rows() - 1)), 0.f);
//            x1 = Math.max(Math.min(x1, (float)(src.cols() - 1)), 0.f);
//            y1 = Math.max(Math.min(y1, (float)(src.rows() - 1)), 0.f);
//
//            rectangle(src, new Point((int)x0, (int)y0), new Point((int)x1, (int)y1), new Scalar(0, 0, 255, 0), 2,LINE_8,
//                    0);
//
//            // 写在目标左上角的内容：类别+置信度
//            final int clsId = classIds.get(idx);
//            String label = names.get(clsId) + ":" + String.format("%.2f%%", confidences.get(idx) * 100f);
//
//            // 计算显示这些内容所需的高度
//            IntPointer baseLine = new IntPointer();
//
//            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
//            y0 = (int) Math.max(y0, labelSize.height());
//
//            // 添加内容到图片上
//            putText(src, label, new Point((int)x0, (int)y0), FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0, 255, 0, 0), 1, LINE_4, false);
//
//            // 释放资源
//            box.releaseReference();
//        }
//
//        // 释放资源
//        indices.releaseReference();
//        confidencesPointer.releaseReference();
//        classIds.releaseReference();
//        confidences.releaseReference();
//        boxes.releaseReference();
//        outs.releaseReference();
//
//        // 图片写到磁盘上
//        imwrite(outImgPath, src);
//
//    }
//}
