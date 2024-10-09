// function navigateToAnotherPage() {
//         var radios = document.getElementsByName('option');
//         var selectedOption = null;
    
//     for (const radio of radios) {
//         if (radio.checked) {
//            alert("Selection process.");
//             selectedOption = radio.value;
//            alert(selectedOption);
//             break;
//         }
//     }
    
//     // Validation check
//     if (selectedOption === null) {
//         //alert("Please select an option.");
//         return false; // Prevent default form submission
//     }
    
//     // Redirect based on selected option
//     switch (selectedOption) {
//         case "Plant Diameter":
//            alert("Selected plant Diameter");
//            window.location.href = "/prediction_diameter";
      
//             break;
//         case "Plant Height":
//            alert("Selected Height.");
//            location.href = "/predict_height";
           
//             break;
//         case "Water pH":
//             alert("Selected pH.");
//            location.href = "/prediction_water_ph";
//             break;
//         case "Water Temperature":
//             //alert("Selected watertemp");
//             location.href = "/prediction_water_temperature";
//             break;
//         default:
//             // Handle unexpected selectedOption value
//             break;
//     }
//      // AJAX POST request to save the selected option
//       var xhr = new XMLHttpRequest();
//       xhr.open("POST", "/save_option", true);
//       //alert("in ajax.");
//       xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
//       xhr.send("option=" + encodeURIComponent(selectedOption));
 
//     return false; // Prevent default form submission
// }

// // Function to handle the prediction when the button is clicked
// function validateForm() 
// {
//   var form = document.getElementById("plantForm");
//   if (!form.checkValidity()) {
//       alert("Please enter valid numeric values.");
//       return false;
//   }
//   return true;
// }
