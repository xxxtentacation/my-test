const fs = require('fs');
const pdf = require('pdf-parse');

let dataBuffer = fs.readFileSync('Computational complexity and exact techniques for scheduling problems with job prohibitions_unlocked.pdf');

pdf(dataBuffer).then(function(data) {
    const text = data.text;
    const startIndex = text.indexOf('2.2 Solving approach');
    if (startIndex !== -1) {
        // Find a reasonable end point, like 2.3 or 3.
        let endIndex = text.indexOf('2.3', startIndex);
        if (endIndex === -1) endIndex = text.indexOf('3.', startIndex);
        if (endIndex === -1) endIndex = startIndex + 3000; // fallback to 3000 chars

        console.log(text.substring(startIndex, endIndex));
    } else {
        console.log('Could not find "2.2 Solving approach" in the PDF.');
    }
});