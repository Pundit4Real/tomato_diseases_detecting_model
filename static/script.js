let body = document.getElementById('body')
let featuresSection = document.getElementById('features-section')
let nav = document.querySelector('nav')
let header = document.querySelector('header')
let scrollToTopBtn = document.getElementById('floating-button')
let input = document.getElementById('image-input')
let preview = document.getElementById('preview')
let form = document.querySelector('form')
let menuWrapper = document.getElementById('menu-wrapper')
let hamburger = document.getElementById('hamburger')

hamburger.addEventListener('click',(e)=>{
    hamburger.classList.toggle('bg-gray-300')
    hamburger.classList.toggle('bg-gray-300')
    menuWrapper.classList.toggle('hidden')
})

window.addEventListener('scroll',()=>{
    if(window.scrollY > nav.offsetHeight){
        nav.classList.remove('bg-opacity-50')
        menuWrapper.classList.remove('bg-opacity-50')
        scrollToTopBtn.classList.remove('hidden')

    }else{
        nav.classList.add('bg-opacity-50')
        menuWrapper.classList.add('bg-opacity-50')

        scrollToTopBtn.classList.add('hidden')
    }
})

document.querySelectorAll('a[href^="#"]').forEach(ele=>{
    ele.addEventListener('click',(e)=>{
        e.preventDefault()

        document.querySelector(ele.getAttribute('href')).scrollIntoView({behavior:'smooth'})
    
    })
})

scrollToTopBtn.addEventListener('click',(e)=>{
    window.scrollTo({top: 0, behavior:'smooth'})
})

input.addEventListener('change',()=>{
    files = input.files;
    
    // FileReader support
    if (FileReader && files && files.length) {
        var fr = new FileReader();
        fr.onload = function () {
            preview.src = fr.result;
        }
        fr.readAsDataURL(files[0]);
    }
    preview.setAttribute('src', input.src)
})

let condition = document.getElementById('condition')
let causes = document.getElementById('causes')
let preventons = document.getElementById('preventions')
let preventionTitle = document.getElementById('preventions-title')
let causesContainer = document.getElementById('causes-container')
form.addEventListener('submit',(e)=>{
    e.preventDefault()
    form.querySelector('button').innerHTML = "<span class='fa fa-spinner fa-spin'></span>" 
    
    fetch('/predict',{body: new FormData(form), method:'POST'}).then((res)=>  res.json()).then((jres)=>{
        document.getElementById('confidence').innerHTML = `<span>${parseFloat(jres['confidence']* 100).toFixed(2)}% confidence level`
        let conditionText =  String(jres['disease']).split("___")[1]
        condition.textContent = conditionText
        if (conditionText == 'Healthy'){
            causesContainer.classList.add('hidden')
            condition.style.color = 'green'
            condition.classList.add('text-green')
            causes.classList.add('hidden')
            preventionTitle.textContent = "Try Also"
        }else{
            causesContainer.classList.remove('hidden')
           
            condition.style.color = 'red'
            condition.classList.remove('text-green')
            causes.classList.remove('hidden')
            preventionTitle.textContent = "Preventions"
        }
        Array.from(causes.children).forEach(ele=> ele.remove())
        Array.from(preventons.children).forEach(ele=> ele.remove())
        if (conditionText != 'Healthy'){
            for(let cause of jres['causes']){
                let p = document.createElement('p')
                p.textContent = cause
                causes.appendChild(p)
            }
            for(let cause of jres['preventions']){
                let p = document.createElement('p')
                p.textContent = cause
                preventons.appendChild(p)
            }
        }else{

            for(let cause of jres['preventions']){
                let p = document.createElement('p')
                p.textContent = cause
                preventons.appendChild(p)
            }
        }
    
        form.querySelector('button').innerHTML = "classify" 
    }).catch((e)=>{
        alert(e)
        form.querySelector('button').innerHTML = "classify" 
    })
   

})