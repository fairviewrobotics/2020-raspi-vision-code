/**
 * Vision code for the high goal (power port)
 */

import org.opencv.core.*;
import org.opencv.imgproc.*;

/* A location of a target in an image */
public class TargetLocation {
  double x, y, w, h, yaw, pitch;

  TargetLocation(double x, double y, double w, double h, double yaw, double pitch) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.yaw = yaw;
    this.pitch = pitch;
  }

  void drawOnImage(Mat image) {
    long img_w = image.width();
    long img_h = image.height();

    Imgproc.rectangle(
      image, 
      new Point(
        (long)(this.x * img_w), 
        (long)(this.y * img_h)
      ),
      new Point(
        (long)((this.x * this.w) * img_w),
        (long)((this.y * this.h) * img_h)
      ),
      new Scalar(0, 0, 255),
      2
    );
  }
}