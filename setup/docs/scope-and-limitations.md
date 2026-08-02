# Scope and Limitations

This study covers a computer vision system that runs an automated badminton
feeder drill and, from the same camera feed, measures how a player performs
during that drill. It works on one half of a court seen by a single camera
mounted on the feeder and pointed across the net, with court geometry set
through a one-time manual calibration of the four corners of the trainee's
half. The system detects and tracks up to two players using a pre-trained
YOLOv8n-pose model, keeps a stable identity for each, and reads their position
to decide which of six target zones they occupy, with optional face recognition
so results attach to a named player. A Roboflow model detects the shuttlecock,
either through a free serverless call or through local weights that run
on-device, and during a drill the feeder fires toward a random zone at an
interval set by the chosen difficulty while the system records whether the
player returned the shuttle in time, scoring results per player and per zone to
reveal weak areas and building a cumulative skill profile across four families
(movement, accuracy, stroke, and posture) that a rule-based rubric scores into
one of five tiers, with a Python backend handling the vision, drill, and scoring
and serving the data to a web dashboard where a coach manages players and reads
the analytics, all written to run later on a Raspberry Pi with a grayscale
OV9281 camera and a feeder wired to the GPIO pins. Within that scope the system
has clear limits: it relies on a single 2D camera, so pose measurements such as
stance width and swing speed are estimated in the image plane rather than in
real units, and a player who is turned side-on or partly hidden produces fewer
usable samples, while posture and stroke metrics are only recorded when the
player is close enough to the camera and each keypoint is confident enough,
leaving thin samples for a player who stays far from the lens. Shuttle detection
is the weakest real-time link, since the free serverless path adds network delay
and holds the loop to a few frames per second, too slow to track a shuttle at
full match speed. 







Face recognition is tuned for a color camera, so its accuracy on the
grayscale OV9281 sensor is untested and the match threshold will likely need
loosening, calibration is manual and static so any camera movement throws off
the court geometry until it is redone, and the reaction-time and swing
thresholds are hand-set and have not been calibrated across players of different
heights and styles, with the skill rubric leaving a player unranked until at
least twenty shots accumulate so that short sessions give partial or no ranking.
Finally, the intended hardware is not fully deployed, as the feeder-firing code
for the Raspberry Pi is still stubbed, so results come from the vision and drill
logic running on a laptop with recorded or webcam footage rather than a complete
feeder-and-Pi setup on court.
