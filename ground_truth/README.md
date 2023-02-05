# Test dataset

![Test dataset](https://github.com/hopemini/activity-clustering-multimodal-ml/blob/main/ground_truth/test_dataset.png)


|Label|Description|
|-----|-----------|
| c1 | c1 displays a type of calendar, including the date picker component. |
| c2 | c2 shows a type of camera. This type is similar to the image type (c21), but this type includes a camera icon. |
| c3 | c3 denotes a type of dialer with 12 grids containing numbers. | 
| c4 | c4 is an input method editor (IME) type with an input function. | 
| c5 | c5 presents a list type containing a list item component. | 
| c6 | c6 is a list type with an on and off switch that does not contain a list item component. The Rico dataset contains specific screens such as c6 and c19, thus, these screens have been categorized separately. | 
| c7 | c7 does not include a list item component. In addition, c7 contains the location icon on the first line. | 
| c8 | c8 displays a type of pinpoint location on a map. | 
| c9 | c9 shows a type of theme containing a page indicator component. | 
| c10 | c10 is a homescreen with app icons. | 
| c11 | c11 is an image list type with grid. | 
| c12 | c12 presents a type of input form with an input component. This type has an editable text field and must receive input to function. This type includes the login activity. | 
| c13 | c13 is similar to c5 in that it has a list of item components. The difference is that c13 is the menu list type, and c5 is displayed on the full screen. | 
| c14 | c14 appears to be similar to that of c13; the main difference between c13 and c14 is the location of the menu list. c13 has a menu list in the upper right, and c14 has a navigation menu in the upper left of the screen. | 
| c15 | c15 is the selection dialog type. This type looks like a popup, except that we have to choose it. | 
| c16 | Depending on the components included in the activity, we divided the text-related categories into three categories: c16, c17, and c18. c16 is a type in which the screen consists of text only. | 
| c17 | c17 is a type consisting of screens with little or no text. | 
| c18 | c18 is a type consisting of a screen with text and images. | 
| c19 | c19 contains a specific screen that is difficult to classify as c18 and c21. | 
| c20 | c20 is a combination of text, images, and buttons. This category type is novel, in that the click of a button will launch a new intent. c20 is divided into r20, r21, r22, and r23. | 
| c21 | c21 provides a type that contains almost any image. c21 is divided into r24, r25, r26, and r27. | 
| c22 | c22 displays a type of ``open and shared with''. c22 is divided into r28 and r29. | 
| c23 | c23 represents any type of popup. c23 is divided into r30, r31, r32, r33, and r34. |
|-----|----------|
| r20 | r20 is an advertisement type with a button to move to another activity at the bottom of the screen.  | 
| r21 | r21 is a type of continued name. This type contains login activity without an input form. | 
| r22 | r22 is a combination of text, images, and buttons, excluding r20 and r21. | 
| r23 | r23 is a combination of text and buttons, except for r21. r23 looks like c16 or c18, but unlike c16 and c18, r23 has a button to launch a new intent. | 
| r24 | r24 is a typical image view type. | 
| r25 | r25 is a type of film. | 
| r26 | r26 is a snapshot. | 
| r27 | r27 is a specific type of the Rico that is difficult to include in r24. | 
| r28 | r28 denotes the ``open with''. | 
| r29 | r29 shows the ``share with''. | 
| r30 | r30 is a typical popup type. | 
| r31 | r31 is a popup with a date picker and number of stepper components. | 
| r32 | r32 has special purposes (automatically login), such as allow, access, and deny. | 
| r33 | r33 is the popup version of r21. | 
| r34 | r34 is a popup type with size of almost full screen. |

* Each category description for C23 and R34. C and R stand for category and revision, respectively. The category order has no special meaning.
