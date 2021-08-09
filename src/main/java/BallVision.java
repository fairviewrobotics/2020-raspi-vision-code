/*
Basic vision algorithm for detecting balls (powercells)
*/

import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;
import org.opencv.objdetect.*;

public class BallVision {

    static class MatContourPair {
        public Mat mat;
        public List<MatOfPoint> contours;

        public MatContourPair(Mat mat, List<MatOfPoint> contours) {
            this.mat = mat;
            this.contours = contours;
        }
    }

    private static TargetLocation target;

    //Load test images, process, and write output images. To-Do: Replace with camera input
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        if (args[0] == "TEST") {
            long hm_images = 18;

            // test using test_inputs, writing to test_outputs
            for (int i = 1; i <= hm_images; i++) {
                Mat input = Imgcodecs.imread("src/test_inputs/BallImage" + i + ".jpg");
                Mat output = decoratePipelineOutput(ballDetectPipeline(input));
                Imgcodecs.imwrite("src/test_outputs/Output Image " + i + ".jpg",
                        output);
            }
            System.out.println("Done");
        } else {
            Mat image = Imgcodecs.imread(args[0]);
            MatContourPair result = ballDetectPipeline(image);

            double yaw = getTargetYaw(
                    findBestBallTarget(result.contours),
                    new Size(result.mat.width(), result.mat.height()),
                    // the following are copied from HighGoalVision.java
                    Math.toRadians(68.5),
                    16.0,
                    9.0
            );
        }
    }

    public static TargetLocation process(Mat mat) {
        MatContourPair matContourPair = ballDetectPipeline(mat);
        Rect best = findBestBallTarget(matContourPair.contours);

        double yaw = getTargetYaw(
                best,
                new Size(matContourPair.mat.width(), matContourPair.mat.height()),
                Math.toRadians(68.5),
                16.0,
                9.0
        );

        return new TargetLocation(
                best.x,
                best.y,
                best.width,
                best.height,
                yaw,
                0 // pitch is irrelevant so just pass 0
        );
    }

    /* calculate yaw or pitch (in radians) */
    static double calcAngle(long val, long centerVal, double focalLen) {
        return Math.atan(((double) (val - centerVal)) / focalLen);
    }

    public static MatContourPair ballDetectPipeline(Mat input) {
        // Resize
        Mat resizeInput = input;
        double resizeWidth = 320.0;
        double resizeHeight = 181.3;
        Size toScale = new Size(resizeWidth, resizeHeight);
        Mat resizedOutput = new Mat();
        Imgproc.resize(resizeInput, resizedOutput, toScale);

        // Blur
        Mat blurInput = resizedOutput;
        double blurRadius = 2.7;//2.7;
        double kernelSize = 2 * (blurRadius + 0.5) + 1;
        Mat blurOutput = new Mat();
        Imgproc.blur(blurInput, blurOutput, new Size(kernelSize, kernelSize));

        // Convert to HSV and Threshold
        Mat hsvThresholdInput = blurOutput;
        double[] hsvThresholdHue = {20.0, 90.0};//{20.0, 180.0};
        double[] hsvThresholdSaturation = {130.0, 255.0};
        double[] hsvThresholdValue = {0.0, 255.0};
        Mat hsvOutput = new Mat();
        Imgproc.cvtColor(hsvThresholdInput, hsvOutput, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvOutput, new Scalar(hsvThresholdHue[0], hsvThresholdSaturation[0], hsvThresholdValue[0]),
                new Scalar(hsvThresholdHue[1], hsvThresholdSaturation[1], hsvThresholdValue[1]), hsvOutput);

        // Find and draw contours
        Imgcodecs.imwrite("HSV Output.jpg", hsvOutput);
        Mat contourInput = hsvOutput;
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        contours.clear();
        Imgproc.findContours(contourInput, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(resizedOutput, contours, -1, new Scalar(0, 0, 255), 1);

        return new MatContourPair(resizedOutput, contours);
    }

    public static Mat decoratePipelineOutput(MatContourPair pipelineOutput) {
        /* Draw guide shapes on the algorithm output for testing purposes.
         * This method is not meant to be run my the robot. */

        // draw a red box around the contours and mark the centers if they exceed an area of 100
        for (MatOfPoint contour : pipelineOutput.contours) {
            double area = Imgproc.contourArea(contour);
            Rect boundRect = Imgproc.boundingRect(contour);
            if (area >= 100) {
                Imgproc.rectangle(
                        pipelineOutput.mat,
                        new Point(boundRect.x, boundRect.y),
                        new Point(boundRect.x + boundRect.width, boundRect.y + boundRect.height),
                        new Scalar(255, 0, 0),
                        5
                );

                Imgproc.drawMarker(pipelineOutput.mat,
                        getRectangleCenter(boundRect),
                        new Scalar(255, 0, 0));
            }
        }

        // draw a green box and marker for the best target
        Rect bestTarget = findBestBallTarget(pipelineOutput.contours);
        Imgproc.rectangle(
                pipelineOutput.mat,
                new Point(bestTarget.x, bestTarget.y),
                new Point(bestTarget.x + bestTarget.width, bestTarget.y + bestTarget.height),
                new Scalar(0, 255, 0),
                5
        );
        Imgproc.drawMarker(pipelineOutput.mat,
                getRectangleCenter(bestTarget),
                new Scalar(0, 255, 0));

        return pipelineOutput.mat;
    }

    public static Rect findBestBallTarget(List<MatOfPoint> contours) {
        double tolerance = 0.3;
        boolean using_tolerances = false;


        // Determine the best bounding rect around the ball contours by width
        long best_centerx = 0;
        long best_centery = 0;
        long largest_width = 0;
        Rect bestBoundRect = new Rect();

        for (MatOfPoint contour : contours) {
            double ballArea = Imgproc.contourArea(contour);

            Rect boundRect = Imgproc.boundingRect(contour);

            // TODO: figure out if these tolerances are useful and/or make useful ones
            boolean in_tolerance = true;
            if (using_tolerances) {
                if (boundRect.height == 0) {
                    continue;
                }

                double ratio = (double) boundRect.width / (double) boundRect.height;
                //System.out.println(ratio);
                in_tolerance = Math.abs(1 - ratio) < tolerance;

                //in_tolerance = boundRect.width >= boundRect.height;*/
            }

            if (ballArea >= 100 && in_tolerance) { // TODO: stop this magic number bullshit
                // create target location
//                Imgproc.rectangle(resizedOutput,
//                        new Point(boundRect.x,boundRect.y),
//                        new Point(boundRect.x+boundRect.width,boundRect.y+boundRect.height),
//                        new Scalar(255,0,0),5);

                long centerx = boundRect.x + boundRect.width / 2;
                long centery = boundRect.y + boundRect.height / 2;

//                System.out.println("Target found!");
//                System.out.println("Center at (" + centerx + ", " + centery + ")");
//                System.out.println("Boundary width: " + boundRect.width);
//                System.out.println("Boundary height: " + boundRect.height);
//                Imgproc.drawMarker(resizedOutput, new Point(centerx,centery), new Scalar(255,0,0));

                // Determine if this is a better target based on width
                if (boundRect.width >= largest_width) {
                    best_centerx = centerx;
                    best_centery = centery;
                    largest_width = boundRect.width;
                    bestBoundRect = boundRect;
                }
            }
        }

        return bestBoundRect;
    }

    public static Point getRectangleCenter(Rect rect) {
        return new Point(
                rect.x + rect.width * 0.5,
                rect.y + rect.height * 0.5
        );
    }

    public static double getTargetYaw(Rect target,
                                      Size resize,
                                      double diagFieldView,
                                      double aspectH,
                                      double aspectV) {
        // calculate camera information
        double aspectDiag = Math.hypot(aspectH, aspectV);

        double fieldViewH = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectH / aspectDiag)) * 2.0;
        double fieldViewV = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectV / aspectDiag)) * 2.0;

        double hFocalLen = resize.width / (2.0 * Math.tan((fieldViewH / 2.0)));

        Point center = getRectangleCenter(target);
        return calcAngle((long) center.x, (long) resize.width / 2, hFocalLen);
    }
}