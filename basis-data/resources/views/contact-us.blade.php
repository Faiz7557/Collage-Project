@extends('layouts.guest')

@section('content')
<div class="container" style="margin-top: 10vh;">
    <div class="row">
        <div class="col-md-7 text-white">
            <h1 class="display-4 fw-bold text-black">Hubungi Kami</h1>
            <p class="lead my-4 text-black">
                Jika Anda memiliki pertanyaan atau masukan, jangan ragu untuk menghubungi kami.
            </p>
            <div class="card glass-morphism-card p-4">
                <form>
                    <div class="mb-3">
                        <label for="name" class="form-label text-black">Nama Lengkap</label>
                        <input type="text" class="form-control" id="name" placeholder="Masukkan nama lengkap Anda">
                    </div>
                    <div class="mb-3">
                        <label for="email" class="form-label text-black">Alamat Email</label>
                        <input type="email" class="form-control" id="email" placeholder="Masukkan alamat email Anda">
                    </div>
                    <div class="mb-3">
                        <label for="message" class="form-label text-black">Pesan Anda</label>
                        <textarea class="form-control" id="message" rows="5" placeholder="Tulis pesan Anda di sini"></textarea>
                    </div>
                    <button type="submit" class="btn btn-success" style="background-color: #4CAF50; border-color: #4CAF50;">Kirim Pesan</button>
                </form>
            </div>
        </div>
    </div>
</div>
@endsection