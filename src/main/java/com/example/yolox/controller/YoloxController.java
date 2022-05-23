package com.example.yolox.controller;

import com.example.yolox.utils.YoloxNet;
import org.bytedeco.opencv.opencv_core.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ResourceLoader;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;


import javax.annotation.PostConstruct;
import java.util.Map;
import java.util.UUID;

import static com.example.yolox.utils.fileHelper.upload;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

@Controller
//@Slf4j
public class YoloxController {
    private final ResourceLoader resourceLoader;

    @Autowired
    public YoloxController(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    @Value("${web.upload-path}")
    private String uploadPath;

    @Value("${opencv.yolox-weights-path}")
    private String weightsPath;

    @Value("${opencv.yolox-coconames-path}")
    private String namesPath;

    @Value("${opencv.yolox-width}")
    private int width;

    @Value("${opencv.yolox-height}")
    private int height;


    // 神经网络
    private YoloxNet yoloxNet;


    @PostConstruct
    private void init() throws Exception {
        yoloxNet = new YoloxNet(weightsPath, namesPath, width, height);
    }

    /**
     * 跳转到文件上传页面
     * @return
     */
    @RequestMapping("index")
    public String toUpload(){
        return "index";
    }


    /**
     *
     * @param file 要上传的文件
     * @return
     */
    @RequestMapping("fileUpload")
    public String fileUpload(@RequestParam("fileName") MultipartFile file, Map<String, Object> map){

        // 文件名称
        String originalFileName = file.getOriginalFilename();

        if (!upload(file, uploadPath, originalFileName)){
            map.put("msg", "上传失败！");
            return "forward:/index";
        }

        // 读取文件到Mat
        Mat src = imread(uploadPath + "/" + originalFileName);
        yoloxNet.detect(src);
        // 新的图片文件名称
        String newFileName = UUID.randomUUID() + ".png";
        // 图片写到磁盘上
        imwrite(uploadPath + "/" + newFileName, src);
        // 文件名
        map.put("fileName", newFileName);
        map.put("msg", "检查结果：");

        return "forward:/index";
    }




    /**
     * 显示单张图片
     * @return
     */
    @RequestMapping("show")
    public ResponseEntity showPhotos(String fileName){
        if (null==fileName) {
            return ResponseEntity.notFound().build();
        }

        try {
            // 由于是读取本机的文件，file是一定要加上的， path是在application配置文件中的路径
            return ResponseEntity.ok(resourceLoader.getResource("file:" + uploadPath + "/" + fileName));
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }
}
