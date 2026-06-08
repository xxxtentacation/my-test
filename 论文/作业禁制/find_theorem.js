const fs = require('fs');
const pdf = require('pdf-parse');

async function findTheorem(file) {
    try {
        let dataBuffer = fs.readFileSync(file);
        let data = await pdf(dataBuffer);
        let text = data.text;
        let index = text.indexOf('Theorem 9');
        if (index !== -1) {
            console.log(`\n--- Found in ${file} ---`);
            console.log(text.substring(Math.max(0, index - 200), index + 2000));
        }
    } catch(e) {
        console.error(e);
    }
}

async function run() {
    await findTheorem('Computational complexity and exact techniques for scheduling problems with job prohibitions_unlocked.pdf');
    await findTheorem('On Solving Travelling Salesman Problem with Vertex.pdf');
}

run();