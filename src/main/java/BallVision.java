/*
Basic vision algorithm for detecting balls (powercells)
*/

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;
import org.opencv.objdetect.*;

public class BallVision{

//Load 6 test images, process, and write 6 output images. To-Do: Replace with camera input
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        for (int i = 1; i<10; i++){
            String image= "src/main/java/BallImage";
            image = image + i + ".jpg";
            Mat input = Imgcodecs.imread(image);
            Mat output = algorithm(input);
            String output_fname = "Output Image " + i + ".jpg";
            Imgcodecs.imwrite(output_fname, output);
        }
        System.out.println("Done");
    }

    /* calculate yaw or pitch (in radians) */
    static double calcAngle(long val, long centerVal, double focalLen) {
        return Math.atan(((double)(val - centerVal)) / focalLen);
    }

    public static Mat algorithm(Mat input) {
        // Resize
        Mat resizeInput = input;
        double resizeWidth = 320.0;
        double resizeHeight = 240.0;
        Size toScale = new Size(resizeWidth, resizeHeight);
        Mat resizedOutput = new Mat();
        Imgproc.resize(resizeInput, resizedOutput, toScale);

        /* calculate camera information
        double aspectDiag = Math.hypot(aspectH, aspectV);

        double fieldViewH = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectH / aspectDiag)) * 2.0;
        double fieldViewV = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectV / aspectDiag)) * 2.0;

        double hFocalLen = resizedOutput.width / (2.0 * Math.tan((fieldViewH / 2.0)));
        double vFocalLen = resizedOutput.height / (2.0 * Math.tan((fieldViewV / 2.0)));

         */
        // Blur
        Mat blurInput = resizedOutput;
		double blurRadius = 2.7;
        double kernelSize = 2 * (blurRadius + 0.5) + 1;
        Mat blurOutput = new Mat();
        Imgproc.blur(blurInput, blurOutput, new Size(kernelSize, kernelSize));
        
        // Convert to HSV and Threshold
        Mat hsvThresholdInput = blurOutput;
		double[] hsvThresholdHue = {20.0, 180.0};
		double[] hsvThresholdSaturation = {130.0, 255.0};
        double[] hsvThresholdValue = {0.0, 255.0};
        Mat hsvOutput = new Mat();
        Imgproc.cvtColor(hsvThresholdInput, hsvOutput, Imgproc.COLOR_BGR2HSV);
		Core.inRange(hsvOutput, new Scalar(hsvThresholdHue[0], hsvThresholdSaturation[0], hsvThresholdValue[0]),
            new Scalar(hsvThresholdHue[1], hsvThresholdSaturation[1], hsvThresholdValue[1]), hsvOutput);

        //Find and draw contours
        Imgcodecs.imwrite("HSV Output.jpg", hsvOutput);
        Mat contourInput = hsvOutput;
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        contours.clear();
        Imgproc.findContours(contourInput, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(resizedOutput, contours, -1, new Scalar(0,0,255),1);
        
        //For each contour found, draw bounding rectangle and marker at center if over a certain size
        double tolerance = 0.3;
        for(MatOfPoint contour : contours) {
            double ballArea = Imgproc.contourArea(contour);

            Rect boundRect = Imgproc.boundingRect(contour);

            if (boundRect.height == 0) {
                continue;
            }

            double ratio = (double)boundRect.width / (double)boundRect.height;
            System.out.println(ratio);
            boolean in_tolerance = Math.abs(1 - ratio) < tolerance;

            //in_tolerance = boundRect.width >= boundRect.height;*/
            if(ballArea>=1000 && in_tolerance){
                Imgproc.rectangle(resizedOutput, new Point(boundRect.x,boundRect.y), new Point(boundRect.x+boundRect.width,boundRect.y+boundRect.height), new Scalar(0,255,0),5);
                long centerx = boundRect.x + boundRect.width / 2;
                long centery = boundRect.y + boundRect.height / 2;
                System.out.println("Target found!");
                System.out.println("Center at (" + centerx + ", " + centery + ")");
                System.out.println("Boundary width: " + boundRect.width);
                System.out.println("Boundary height: " + boundRect.height);
                Imgproc.drawMarker(resizedOutput, new Point(centerx,centery), new Scalar(0,255,0));

                double focalLen = 7.0; // estimated focal length of Edward's camera
                double ballSize = 177.8; // 7 inches in mm
                double sensorHeight = 18.0;
                double distance = (focalLen * ballSize * resizeWidth) / ((double)boundRect.width * sensorHeight);
                System.out.println("Distance from Camera: " + distance + "mm");
            }

        }
        return resizedOutput;
    }

}
