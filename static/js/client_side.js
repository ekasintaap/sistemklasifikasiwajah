$(document).ready(function(){
  
  // -[Animasi Scroll]---------------------------
  
  $(".navbar a, footer a[href='#halamanku']").on('click', function(event) {
    if (this.hash !== "") {
      event.preventDefault();
      var hash = this.hash;
      $('html, body').animate({
        scrollTop: $(hash).offset().top
      }, 900, function(){
        window.location.hash = hash;
      });
    } 
  });
  
  $(window).scroll(function() {
    $(".slideanim").each(function(){
      var pos = $(this).offset().top;
      var winTop = $(window).scrollTop();
        if (pos < winTop + 600) {
          $(this).addClass("slide");
        }
    });
  });
  
  // -[Prediksi Model]---------------------------
  
  // Fungsi untuk memanggil API ketika tombol prediksi ditekan
  $("#prediksi_submit").click(function(e) {
    e.preventDefault();
	
	// Get File Gambar yg telah diupload pengguna
    var file_data = $('#input_gambar').prop('files')[0];   
    var pics_data = new FormData();                  
    pics_data.append('file', file_data);
	
  

	// Panggil API dengan timeout 1 detik (1000 ms)
    setTimeout(function() {
	  try {
			$.ajax({
				url         : "/api/deteksi",
				type        : "POST",
				data        : pics_data,
				processData : false,
				contentType : false,
				success     : function(res){
					// Ambil hasil prediksi dan path gambar yang diprediksi dari API
					res_data_prediksi   = res['prediksi']
					res_gambar_prediksi = res['gambar_prediksi']
					
					// Tampilkan hasil prediksi ke halaman web
					generate_prediksi(res_data_prediksi, res_gambar_prediksi); 
					mengatasi(res_data_prediksi);
					jenis(res_data_prediksi);
			  }
			});
		}
		catch(e) {
			// Jika gagal memanggil API, tampilkan error di console
			console.log("Gagal !");
			console.log(e);
		} 
    }, 1000)  
  })
   
  // Fungsi untuk menampilkan hasil prediksi model
  function generate_prediksi(data_prediksi, image_prediksi) {
	var str="";
	let elm_hasil_prediksi = document.getElementById("gambar_prediksi");
	
	elm_hasil_prediksi.setAttribute("src", image_prediksi);
	str = "<h3 style='color:#99627A'> Wajah anda terdeteksi " + data_prediksi + "</h3>";

	console.log("Cek Image Prediksi:", image_prediksi);
	// if (image_prediksi == "(none)" || !image_prediksi) {
	// 	str += "<h3 style='color:#99627A; font-size:18px'> Hasil Prediksi </h3>";
	// 	str += "<br>";
	// 	str += "<img src='/static/images/uploads/default-img.jpg' class='img-border-radius' width='200'></img>";
	// 	str += "<h4 style='color:#99627A'>Silahkan masukkan file gambar (.jpg)</h4>";
	// $("#hasil_prediksi").html(str);
	// } else {
	// 	str += "<p style='color:#99627A; font-size:18px'>Hasil Prediksi </p>";
	// 	str += "<br>";
	// 	str += "<img src='" + image_prediksi + "' class='img-border-radius' width='200'></img>";
	// 	str += "<h3 style='color:#99627A'> Wajah anda terdeteksi " + data_prediksi + "</h3>";
	// }
	
	$("#hasil_prediksi").html(str);
	
  } 

  function mengatasi (data_prediksi) {
	var str="";

	if(data_prediksi == "Acne"){
		str += "<h4 class='font-mengatasi'>1.<span style='color: red;'>	Membersihkan kulit secara teratur: </span> Rajin membersihkan kulit wajah dua kali sehari dengan pembersih yang lembut dapat membantu menghilangkan minyak berlebih, kotoran, dan bakteri yang dapat menyebabkan jerawat. <br>  2.<span style='color: red;'>Hindari menyentuh wajah dengan tangan yang kotor: </span> Tangan yang kotor dapat mengandung kuman dan bakteri yang dapat memperburuk jerawat. Hindarilah menyentuh wajah secara berlebihan dan pastikan tangan dalam keadaan bersih saat menyentuh wajah. <br>  3.<span style='color: red;'>Gunakan produk perawatan kulit yang sesuai: </span> Pilih produk perawatan kulit yang cocok untuk jenis kulit Anda. Hindari produk yang mengandung bahan iritan atau komedogenik yang dapat menyumbat pori-pori dan memperburuk jerawat. <br>   4.Hindari memencet atau menggaruk jerawat: </span> Memencet atau menggaruk jerawat dapat menyebabkan peradangan yang lebih parah, meninggalkan bekas luka, dan memperluas infeksi. <br>   5. <span style='color: red;'>Gunakan produk yang mengandung bahan aktif seperti asam salisilat atau benzoyl peroxide:   Bahan-bahan ini dapat membantu mengurangi produksi minyak berlebih, mengurangi peradangan, dan membunuh bakteri penyebab jerawat. <br>   6.Jaga pola makan sehat: Konsumsi makanan yang seimbang, kaya akan nutrisi, dan rendah gula serta makanan berlemak dapat membantu menjaga kesehatan kulit. <br>   7.Hindari paparan sinar matahari berlebih: Paparan sinar matahari berlebih dapat meningkatkan produksi minyak dan peradangan kulit. Gunakan tabir surya dengan SPF yang sesuai dan hindari terlalu lama berada di bawah sinar matahari langsung. <br>   8.Kurangi stres: Stres dapat memicu produksi hormon yang dapat meningkatkan produksi minyak dan menyebabkan jerawat. Cari cara untuk mengelola stres, seperti dengan olahraga, meditasi, atau kegiatan yang menenangkan <br>   9.Jaga kebersihan benda yang sering bersentuhan dengan wajah: Benda-benda seperti handuk, bantal, atau telepon seluler dapat menjadi sarang bakteri yang dapat memperburuk jerawat. Pastikan untuk menjaga kebersihan benda-benda ini secara teratur. <br>   10.Konsultasikan dengan dokter atau ahli dermatologi: Jika jerawat Anda parah atau tidak kunjung membaik dengan perawatan mandiri, sebaiknya konsultasikan dengan dokter atau ahli dermatologi. Mereka dapat memberikan saran dan perawatan yang lebih khusus sesuai dengan kondisi kulit Anda.</h4>";
	}else if(data_prediksi == "Flek hitam"){
		str += "<h4>1. <span style='color: red;'> Gunakan tabir surya: </span> Flek hitam seringkali disebabkan oleh paparan sinar matahari yang berlebihan. Menggunakan tabir surya dengan SPF yang cukup setiap hari dapat membantu melindungi kulit dari sinar UV dan mencegah munculnya flek hitam. <br>   2.<span style='color: red;'>Hindari paparan sinar matahari secara langsung: </span> Selain menggunakan tabir surya, hindari paparan sinar matahari langsung terutama pada jam-jam matahari paling terik, yaitu antara pukul 10 pagi hingga 4 sore. <br>   3.<span style='color: red;'>Gunakan krim pemutih: </span> Krim pemutih yang mengandung bahan seperti hydroquinone, kojic acid, asam azelaic, atau asam retinoat dapat membantu mengurangi flek hitam dengan menghambat produksi melanin. <br>   4.<span style='color: red;'>Perawatan laser:  </span> Terapi laser seperti laser pigmentasi atau laser CO2 fraksional dapat membantu menghilangkan flek hitam pada wajah. Namun, perlu konsultasi dengan dokter atau ahli dermatologi untuk menentukan jenis perawatan yang sesuai untuk kondisi Anda. <br>   5.<span style='color: red;'>Menggunakan peeling kimia: </span> Peeling kimia dengan asam glikolat, asam salisilat, atau asam mandelat dapat membantu mengelupas lapisan kulit yang terkena flek hitam dan merangsang pertumbuhan sel kulit baru yang lebih cerah. <br>   6.<span style='color: red;'>Rutin membersihkan wajah: </span>Membersihkan wajah secara teratur dan menyeluruh dapat membantu menghilangkan kotoran, minyak, dan sel kulit mati yang dapat memperburuk penampilan flek hitam. <br>   7.<Menggunakan serum atau krim dengan vitamin C: Vitamin C memiliki sifat antioksidan dan dapat membantu mencerahkan kulit serta mengurangi tampilan flek hitam. Gunakan serum atau krim yang mengandung vitamin C secara teratur. <br>    8.	Menggunakan masker wajah alami: Beberapa bahan alami seperti jus lemon, yogurt, madu, atau lidah buaya memiliki sifat pemutih alami dan dapat membantu mengurangi flek hitam. Gunakan masker wajah alami ini secara teratur. <br>   9.	Hindari penggunaan kosmetik yang berat: Penggunaan kosmetik yang berat, terutama yang mengandung bahan berbahaya atau iritan, dapat memperburuk kondisi flek hitam. Pilihlah kosmetik yang ringan dan bebas minyak.    10.	Gaya hidup sehat: Gaya hidup sehat, termasuk konsumsi makanan bergizi, tidur yang cukup, dan mengelola stres, dapat membantu menjaga kesehatan kulit dan mengurangi risiko ,munculnya flek hitam. </h4>"
  } else if(data_prediksi == "Healthy"){
		str += "<h4> 1.<span style='color: red;'>Rutin membersihkan wajah: </span> Cuci wajah secara teratur, setidaknya dua kali sehari, menggunakan pembersih yang lembut dan sesuai dengan jenis kulit Anda. Ini akan membantu menghilangkan kotoran, minyak berlebih, dan sisa-sisa makeup yang dapat menyumbat pori-pori. <br> 2.<span style='color: red;'>Menjaga kelembapan kulit:</span> Gunakan pelembap yang cocok untuk jenis kulit Anda setelah mencuci wajah. Ini akan membantu menjaga kelembapan alami kulit dan mencegah kulit menjadi kering dan kusam. <br> 3.<span style='color: red;'>Melindungi kulit dari sinar matahari: </span> Gunakan tabir surya dengan SPF yang cukup setiap kali Anda berada di luar ruangan, bahkan saat cuaca mendung. Paparan sinar matahari yang berlebihan dapat menyebabkan penuaan dini, flek hitam, dan kerusakan kulit lainnya. <br> 4.<span style='color: red;'>Hindari merokok dan konsumsi alkohol berlebihan: </span> Merokok dan konsumsi alkohol berlebihan dapat merusak kesehatan kulit, menyebabkan keriput, garis-garis halus, dan penampilan kusam. Hindari kebiasaan ini untuk menjaga kondisi kulit yang sehat. <br>  5.<span style='color: red;'>Konsumsi makanan sehat: </span> Pola makan yang seimbang dengan asupan nutrisi yang cukup dapat berdampak positif pada kondisi kulit. Konsumsilah makanan yang kaya akan vitamin, mineral, dan antioksidan seperti buah-buahan, sayuran, biji-bijian, dan ikan berlemak. <br> 6.<span style='color: red;'>Minum cukup air: </span>Pastikan Anda terhidrasi dengan baik dengan meminum cukup air setiap hari. Air membantu menjaga kelembapan kulit dan menghilangkan racun dari tubuh. <br>  7. 	Hindari stres berlebihan: Stres dapat mempengaruhi kesehatan kulit. Cari cara untuk mengelola stres seperti dengan berolahraga, meditasi, atau aktivitas relaksasi lainnya. <br> 8.	Tidur yang cukup: Pastikan Anda mendapatkan tidur yang cukup setiap malam. Tidur yang cukup membantu regenerasi sel-sel kulit dan memperbaiki kerusakan yang terjadi selama hari.<br> 9.	Hindari sentuhan berlebih pada wajah: Hindari menyentuh wajah secara berlebihan, terutama dengan tangan yang kotor. Tangan dapat mengandung kuman dan bakteri yang dapat menyebabkan infeksi kulit. <br> 10.	Rutin kunjungi ahli dermatologi: Konsultasikan dengan ahli dermatologi secara rutin untuk pemeriksaan kulit dan perawatan yang sesuai. Ahli dermatologi dapat memberikan saran dan perawatan khusus untuk menjaga kondisi kulit yang sehat.</h4>"
  }else if(data_prediksi == "Rosacea"){
		str += "<h4> 1.<span style='color: red;'>Konsultasi dengan dokter atau ahli dermatologi: </span> Anda mengalami gejala rosacea pada wajah, sangat penting untuk berkonsultasi dengan dokter atau ahli dermatologi. Mereka dapat mendiagnosis kondisi dengan tepat dan memberikan penanganan yang sesuai. <br> 2.	Hindari pemicu rosacea: Identifikasi faktor pemicu yang memicu flare-up rosacea pada wajah Anda, seperti paparan sinar matahari, makanan pedas, alkohol, stres, atau perubahan suhu. Upayakan untuk menghindari atau mengurangi paparan terhadap pemicu-pemicu ini.  <br> 3.	Gunakan tabir surya dengan SPF: Paparan sinar matahari dapat memperburuk gejala rosacea. Gunakan tabir surya dengan SPF yang tinggi dan lindungi wajah Anda dari sinar matahari dengan menggunakan topi atau payung saat berada di luar ruangan.  <br> 4.	Gunakan produk perawatan kulit yang lembut: Pilih produk perawatan kulit yang dirancang khusus untuk kulit sensitif atau rosacea. Hindari penggunaan produk yang mengandung bahan iritan seperti alkohol, pewangi, atau detergen yang keras. <br> 5.	Gunakan pelembap yang lembut: Menggunakan pelembap yang cocok untuk kulit sensitif dapat membantu menjaga hidrasi kulit dan mengurangi gejala kemerahan dan kering pada rosacea. <br> 6.	Hindari penggunaan produk perawatan yang mengandung scrub atau eksfolian yang kasar: Penggunaan produk perawatan yang agresif dapat memperburuk iritasi pada kulit rosacea. Pilih produk yang lembut dan tidak mengandung partikel-partikel kasar. <br> 7.	Gunakan make-up yang sesuai: Jika Anda menggunakan make-up, pilihlah produk yang dirancang khusus untuk kulit sensitif atau rosacea. Gunakan foundation yang ringan, hindari produk yang mengandung pewarna atau bahan iritan, dan bersihkan make-up secara menyeluruh setelah digunakan. <br> 8.	Hindari penggunaan air panas: Air panas dapat memicu flare-up pada rosacea. Cuci wajah Anda dengan air hangat atau suhu yang nyaman, dan hindari air yang terlalu panas saat mandi atau mencuci wajah. <br> 9.	Gunakan kompres dingin: Jika kulit Anda sedang meradang atau terasa panas, gunakan kompres dingin atau handuk yang dibasahi dengan air dingin untuk meredakan peradangan dan kemerahan. <br> 10.	Terapi laser atau fotodinamik: Dalam beberapa kasus yang lebih parah, dokter atau ahli dermatologi mungkin merekomendasikan terapi laser atau fotodinamik untuk mengurangi gejala rosacea. Terapi ini dapat membantu mengurangi kemerahan, pembuluh darah yang terlihat, dan peradangan pada kulit. </h4>"
  }else {
		str += "<h4>1.<span style='color: red;'>	Konsultasi dengan dokter: </span> Jika Anda mengalami panu pada wajah, sangat disarankan untuk berkonsultasi dengan dokter atau ahli dermatologi. Mereka dapat mendiagnosis kondisi dengan tepat dan memberikan penanganan yang sesuai. <br> 2. <span style='color: red;'>	Obat antifungal topikal: </span> Dokter dapat meresepkan obat antifungal topikal seperti krim atau salep yang mengandung bahan aktif seperti ketoconazole, miconazole, atau clotrimazole. Obat ini dapat membantu menghilangkan infeksi jamur penyebab panu.  <br> 3.<span style='color: red;'>Penggunaan sampo antifungal: </span> Jika panu juga ada di area kulit kepala atau di sekitar rambut, dokter mungkin merekomendasikan penggunaan sampo antifungal yang mengandung bahan seperti selenium sulfida atau ketoconazole.  <br> 4.<span style='color: red;'> Perawatan sistemik: Jika infeksi panu cukup luas atau tidak merespon terhadap pengobatan topikal, dokter dapat meresepkan obat antifungal dalam bentuk tablet atau kapsul. Pengobatan sistemik ini seringkali digunakan dalam kasus-kasus yang lebih parah.  <br> 5.	Hindari faktor pemicu: Beberapa faktor seperti kelembaban, panas, atau penggunaan kosmetik yang berat dapat memperburuk panu. Dokter mungkin menyarankan untuk menghindari faktor-faktor ini untuk membantu mengatasi panu dengan lebih baik.  <br> 6.	Kebersihan dan perawatan kulit: Menjaga kebersihan kulit wajah dan menjaga kelembapan kulit dapat membantu mengurangi risiko infeksi dan mempercepat pemulihan. Gunakan pembersih wajah yang lembut dan hindari menggosok atau menggaruk kulit yang terinfeksi.  <br> 7.	Jangan berbagi barang-barang pribadi: Untuk mencegah penyebaran infeksi jamur, hindari berbagi barang-barang pribadi seperti handuk, pakaian, atau peralatan makeup dengan orang lain.  <br> 8.	Perhatikan kebersihan pakaian dan seprai: Cuci pakaian, seprai, dan handuk secara teratur dengan air panas untuk membunuh jamur yang mungkin ada di dalamnya. <br> 9.	Hindari perawatan wajah yang agresif: Penggunaan bahan atau perawatan wajah yang agresif seperti scrub atau astringen yang keras dapat memperburuk iritasi dan peradangan. Gunakan produk perawatan yang lembut dan sesuai dengan kulit Anda.  <br> 10.	Patuhi petunjuk pengobatan: Pastikan untuk mengikuti petunjuk dan rekomendasi pengobatan yang diberikan oleh dokter atau ahli dermatologi. Gunakan obat sesuai dengan dosis dan durasi yang ditentukan untuk memastikan pengobatan yang efektif. </h4>"
  }

  $("#mengatasi").html(str);
}

  function jenis (data_prediksi){
	var str = " ";

	if (data_prediksi == "Acne"){
		str +="<h2> CARA MENGATASI PENYAKIT JERAWAT</h2>"
	}else if (data_prediksi == "Flek Hitam") {
		str +="<h2>  CARA MENGATASI FLEK HITAM</h2>"
	}else if (data_prediksi == "Healthy"){
		str +="<h2> CARA MENJAGA AGAR KULIT TETAP SEHAT</h2>"
	}else if(data_prediksi == "osacea"){
		str +="<h2>  CARA MENGATASI PENYAKIT KEMERAHAN </h2>"
	}else{
		str +="<h2>CARA MENGATASI PENYAKIT PANU</h2>"
	}

	$("#jenis").html(str);
  }
  
});
