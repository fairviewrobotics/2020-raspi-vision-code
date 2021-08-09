/**
 * Vision code for the high goal (power port)
 * 
 * IMPORTANT: to set exposure properly
 * v4l2-ctl -c exposure_auto=1 -c exposure_absolute=10 -d /dev/videoX
 */

import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.core.*;
import org.opencv.core.Core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;
import org.opencv.objdetect.*;

import java.util.List;
import java.util.ArrayList;
import java.lang.Math;
import java.util.Collections;

public class HighGoalVision {

  /* calculate yaw or pitch (in radians) */
  static double calcAngle(long val, long centerVal, double focalLen) {
    return Math.atan(((double)(val - centerVal)) / focalLen);
  }

  public static List<TargetLocation> highGoalDetect(
    Mat image,
    Size resize,
    Scalar lowHsv,
    Scalar highHsv,
    long blurRadius,
    long reblurRadius,
    double minAreaRatio,
    double diagFieldView,
    double aspectH,
    double aspectV
  ) {

    /* calculate camera information */
    double aspectDiag = Math.hypot(aspectH, aspectV);

    double fieldViewH = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectH / aspectDiag)) * 2.0;
    double fieldViewV = Math.atan(Math.tan(diagFieldView / 2.0) * (aspectV / aspectDiag)) * 2.0;

    double hFocalLen = resize.width / (2.0 * Math.tan((fieldViewH / 2.0)));
    double vFocalLen = resize.height / (2.0 * Math.tan((fieldViewV / 2.0)));

    /* resize */
    Mat imageRs = new Mat();
    Imgproc.resize(image, imageRs, resize);

    /* blur */
    Mat imageBlur = new Mat();
    Imgproc.blur(imageRs, imageBlur, new Size(blurRadius, blurRadius));

    /* convert to hsv */
    Mat imageHsv = new Mat();
    Imgproc.cvtColor(imageBlur, imageHsv, Imgproc.COLOR_BGR2HSV);

    /* mask hsv to specified range */
    Mat imageMask = new Mat();
    Core.inRange(imageHsv, new Scalar(53, 213, 100), new Scalar(53, 213, 100), imageMask);

    /* blur and re-threshold image (further noise elimination) */
    Imgproc.blur(imageMask, imageBlur, new Size(reblurRadius, reblurRadius));
    Imgproc.threshold(imageBlur, imageMask, 50, 255, Imgproc.THRESH_BINARY);

    /* find contours */
    List<MatOfPoint> contours = new ArrayList<>();
    Mat contourHierarchy = new Mat();

    Imgproc.findContours(imageMask, contours, contourHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_TC89_KCOS);

    /* sort by size (largest -> smallest) */
    Collections.sort(contours, (MatOfPoint cnt1, MatOfPoint cnt2) -> -Double.compare(Imgproc.contourArea(cnt1), Imgproc.contourArea(cnt2)));

    ArrayList<TargetLocation> targets = new ArrayList<>();

    for(MatOfPoint cnt : contours) {
      double hullArea = Imgproc.contourArea(cnt);
      double imageArea = resize.width * resize.height;
      /* check target is largest than minimum size */
      if(hullArea / imageArea >= minAreaRatio) {
        /* find bounding rect on target */
        Rect boundRect = Imgproc.boundingRect(cnt);
        /* because vision target is just on the lower half of the real target, adjust box to include top half */
        boundRect.y -= boundRect.height;
        boundRect.height *= 1.9;

        Imgproc.rectangle(imageRs, new Point(boundRect.x, boundRect.y), new Point(boundRect.x + boundRect.width, boundRect.y + boundRect.height), new Scalar(0, 0, 255), 2);
        Imgcodecs.imwrite("dst.jpg", imageRs);


        /* find center of target */
        long cx = boundRect.x + boundRect.width / 2;
        long cy = boundRect.y + boundRect.height / 2;

        /* calculate yaw + pitch offset from center (in radians) */
        double yaw = calcAngle(cx, (long)resize.width/2, hFocalLen);
        double pitch = calcAngle(cy, (long)resize.height/2, vFocalLen);

        /* construct target information */
        targets.add(new TargetLocation(
          ((double)boundRect.x) / ((double)resize.width),
          ((double)boundRect.y) / ((double)resize.height),
          ((double)boundRect.width) / ((double) resize.width),
          ((double)boundRect.height) / ((double)resize.height),
          yaw, 
          pitch));
      }
    }

    return targets;
  }

  public static void main(String[] args) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    Mat image = Imgcodecs.imread(args[0]);

    List<TargetLocation> targets = highGoalDetect(image, new Size(320, 240), new Scalar(53, 213, 100), new Scalar(100, 255, 255), 5, 5, 0.001, Math.toRadians(68.5), 16.0, 9.0);
  }
}